package postgres

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"

	"github.com/MainfluxLabs/mainflux/consumers/alarms"
	"github.com/MainfluxLabs/mainflux/pkg/apiutil"
	"github.com/MainfluxLabs/mainflux/pkg/dbutil"
	"github.com/MainfluxLabs/mainflux/pkg/errors"
	"github.com/gofrs/uuid"
	"github.com/jackc/pgerrcode"
	"github.com/jackc/pgx/v5/pgconn"
)

var _ alarms.AlarmRepository = (*alarmRepository)(nil)

type alarmRepository struct {
	db dbutil.Database
}

func NewAlarmRepository(db dbutil.Database) alarms.AlarmRepository {
	return &alarmRepository{
		db: db,
	}
}

func (ar *alarmRepository) Save(ctx context.Context, alarms ...alarms.Alarm) error {
	tx, err := ar.db.BeginTxx(ctx, nil)
	if err != nil {
		return errors.Wrap(errors.ErrCreateEntity, err)
	}

	q := `INSERT INTO alarms (id, thing_id, group_id, subtopic, protocol, payload, created)
	      VALUES (:id, :thing_id, :group_id, :subtopic, :protocol, :payload, :created);`

	for _, alarm := range alarms {
		dbAlarm, err := toDBAlarm(alarm)
		if err != nil {
			tx.Rollback()
			return errors.Wrap(errors.ErrCreateEntity, err)
		}

		if _, err := tx.NamedExecContext(ctx, q, dbAlarm); err != nil {
			tx.Rollback()
			pgErr, ok := err.(*pgconn.PgError)
			if ok {
				switch pgErr.Code {
				case pgerrcode.InvalidTextRepresentation:
					return errors.Wrap(errors.ErrMalformedEntity, err)
				case pgerrcode.UniqueViolation:
					return errors.Wrap(errors.ErrConflict, err)
				case pgerrcode.StringDataRightTruncationWarning:
					return errors.Wrap(errors.ErrMalformedEntity, err)
				}
			}
			return errors.Wrap(errors.ErrCreateEntity, err)
		}
	}

	if err = tx.Commit(); err != nil {
		return errors.Wrap(errors.ErrCreateEntity, err)
	}
	return nil
}

func (ar *alarmRepository) RetrieveByID(ctx context.Context, id string) (alarms.Alarm, error) {
	q := `SELECT id, thing_id, group_id, subtopic, protocol, payload, created FROM alarms WHERE id = $1;`

	var dba dbAlarm
	if err := ar.db.QueryRowxContext(ctx, q, id).StructScan(&dba); err != nil {
		pgErr, ok := err.(*pgconn.PgError)
		if err == sql.ErrNoRows || ok && pgerrcode.InvalidTextRepresentation == pgErr.Code {
			return alarms.Alarm{}, errors.Wrap(errors.ErrNotFound, err)
		}
		return alarms.Alarm{}, errors.Wrap(errors.ErrRetrieveEntity, err)
	}

	return toAlarm(dba)
}

func (ar *alarmRepository) RetrieveByThing(ctx context.Context, thingID string, pm apiutil.PageMetadata) (alarms.AlarmsPage, error) {
	if _, err := uuid.FromString(thingID); err != nil {
		return alarms.AlarmsPage{}, errors.Wrap(errors.ErrNotFound, err)
	}

	oq := dbutil.GetOrderQuery(pm.Order)
	dq := dbutil.GetDirQuery(pm.Dir)
	olq := dbutil.GetOffsetLimitQuery(pm.Limit)
	p, pq, err := dbutil.GetPayloadQuery(pm.Payload)
	if err != nil {
		return alarms.AlarmsPage{}, errors.Wrap(errors.ErrRetrieveEntity, err)
	}

	thingFilter := "thing_id = :thing_id"
	whereClause := dbutil.BuildWhereClause(thingFilter, pq)

	q := fmt.Sprintf(`SELECT id, thing_id, group_id, subtopic, protocol, payload, created 
	                  FROM alarms %s ORDER BY %s %s %s;`, whereClause, oq, dq, olq)
	qc := fmt.Sprintf(`SELECT COUNT(*) FROM alarms %s;`, whereClause)

	params := map[string]interface{}{
		"thing_id": thingID,
		"limit":    pm.Limit,
		"offset":   pm.Offset,
		"payload":  p,
	}

	return ar.retrieve(ctx, q, qc, params)
}

func (ar *alarmRepository) RetrieveByGroup(ctx context.Context, groupID string, pm apiutil.PageMetadata) (alarms.AlarmsPage, error) {
	if _, err := uuid.FromString(groupID); err != nil {
		return alarms.AlarmsPage{}, errors.Wrap(errors.ErrNotFound, err)
	}

	oq := dbutil.GetOrderQuery(pm.Order)
	dq := dbutil.GetDirQuery(pm.Dir)
	olq := dbutil.GetOffsetLimitQuery(pm.Limit)

	p, pq, err := dbutil.GetPayloadQuery(pm.Payload)
	if err != nil {
		return alarms.AlarmsPage{}, errors.Wrap(errors.ErrRetrieveEntity, err)
	}

	groupFilter := "group_id = :group_id"
	whereClause := dbutil.BuildWhereClause(groupFilter, pq)

	q := fmt.Sprintf(`SELECT id, thing_id, group_id, subtopic, protocol, payload, created 
	                  FROM alarms %s ORDER BY %s %s %s;`, whereClause, oq, dq, olq)
	qc := fmt.Sprintf(`SELECT COUNT(*) FROM alarms %s;`, whereClause)

	params := map[string]interface{}{
		"group_id": groupID,
		"limit":    pm.Limit,
		"offset":   pm.Offset,
		"payload":  p,
	}

	return ar.retrieve(ctx, q, qc, params)
}

func (ar *alarmRepository) Remove(ctx context.Context, ids ...string) error {
	for _, id := range ids {
		dba := dbAlarm{ID: id}
		q := `DELETE FROM alarms WHERE id = :id;`

		_, err := ar.db.NamedExecContext(ctx, q, dba)
		if err != nil {
			return errors.Wrap(errors.ErrRemoveEntity, err)
		}
	}

	return nil
}

func (ar *alarmRepository) retrieve(ctx context.Context, query, cquery string, params map[string]interface{}) (alarms.AlarmsPage, error) {
	rows, err := ar.db.NamedQueryContext(ctx, query, params)
	if err != nil {
		return alarms.AlarmsPage{}, errors.Wrap(errors.ErrRetrieveEntity, err)
	}
	defer rows.Close()

	var items []alarms.Alarm
	for rows.Next() {
		var dbAlarm dbAlarm
		if err := rows.StructScan(&dbAlarm); err != nil {
			return alarms.AlarmsPage{}, errors.Wrap(errors.ErrRetrieveEntity, err)
		}

		alarm, err := toAlarm(dbAlarm)
		if err != nil {
			return alarms.AlarmsPage{}, errors.Wrap(errors.ErrRetrieveEntity, err)
		}

		items = append(items, alarm)
	}

	total, err := dbutil.Total(ctx, ar.db, cquery, params)
	if err != nil {
		return alarms.AlarmsPage{}, errors.Wrap(errors.ErrRetrieveEntity, err)
	}

	page := alarms.AlarmsPage{
		Alarms: items,
		PageMetadata: apiutil.PageMetadata{
			Total:  total,
			Offset: params["offset"].(uint64),
			Limit:  params["limit"].(uint64),
		},
	}

	return page, nil
}

type dbAlarm struct {
	ID       string `db:"id"`
	ThingID  string `db:"thing_id"`
	GroupID  string `db:"group_id"`
	Subtopic string `db:"subtopic"`
	Protocol string `db:"protocol"`
	Payload  []byte `db:"payload"`
	Created  int64  `db:"created"`
}

func toDBAlarm(alarm alarms.Alarm) (dbAlarm, error) {
	payload, err := json.Marshal(alarm.Payload)
	if err != nil {
		return dbAlarm{}, err
	}

	return dbAlarm{
		ID:       alarm.ID,
		ThingID:  alarm.ThingID,
		GroupID:  alarm.GroupID,
		Subtopic: alarm.Subtopic,
		Protocol: alarm.Protocol,
		Payload:  payload,
		Created:  alarm.Created,
	}, nil
}

func toAlarm(dbAlarm dbAlarm) (alarms.Alarm, error) {
	var payload map[string]interface{}
	if err := json.Unmarshal(dbAlarm.Payload, &payload); err != nil {
		return alarms.Alarm{}, errors.Wrap(errors.ErrMalformedEntity, err)
	}

	return alarms.Alarm{
		ID:       dbAlarm.ID,
		ThingID:  dbAlarm.ThingID,
		GroupID:  dbAlarm.GroupID,
		Subtopic: dbAlarm.Subtopic,
		Protocol: dbAlarm.Protocol,
		Payload:  payload,
		Created:  dbAlarm.Created,
	}, nil
}
