// Copyright (c) Mainflux
// SPDX-License-Identifier: Apache-2.0

package readers

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"time"

	"gonum.org/v1/gonum/mat"
)

// ARXDataPoint represents one row of the hourly timber moisture dataset.
// JSON tags match the example payload format.
type ARXDataPoint struct {
	Time      string  `json:"Time"`
	AmbientRH float64 `json:"Ambient RH [%]"`
	AmbientT  float64 `json:"Ambient T [%]"`
	SensorMCA float64 `json:"Sensor MC A (38mm) [%]"`
	SensorMCB float64 `json:"Sensor MC B (48mm) [%]"`
}

// First-order ARX(1) coefficients from the paper (Final Report - ARX Modelling of Timber Moisture):
// MC_A(t) ≈ β₀ + φ₁·MC_A(t-4h) + γ₁·RH(t-4h) + δ₃·Temp(t-12h)
const (
	arxBeta0  = 0.20
	arxPhi1   = 0.957
	arxGamma1 = 0.0056
	arxDelta3 = 0.00051
)

func execReincarnateAlgo(ctx context.Context, rs *readersService, realDeviceID string) (JSONMessagesPage, error) {
	// Get all datapoints from source device and convert them to ARXDataPoint
	sourceMsgs, err := rs.json.Retrieve(ctx, JSONPageMetadata{
		Publisher: realDeviceID,
		Limit:     10000,
	})

	if err != nil {
		return JSONMessagesPage{}, err
	}

	var datapoints = make([]ARXDataPoint, 0, sourceMsgs.Total)
	for _, msg := range sourceMsgs.Messages {
		payload := msg.(map[string]any)["payload"].(map[string]any)

		datapoints = append(datapoints, ARXDataPoint{
			Time:      payload["Time"].(string),
			AmbientRH: payload["Ambient RH [%]"].(float64),
			AmbientT:  payload["Ambient T [%]"].(float64),
			SensorMCA: payload["Sensor MC A (38mm) [%]"].(float64),
			SensorMCB: payload["Sensor MC B (48mm) [%]"].(float64),
		})

	}

	fmt.Println("Reincarnate Hack: computing algo...")

	result, err := ARXMCAPredictOLS(datapoints)
	// result, err := ARXMCAPredict(datapoints)
	if err != nil {
		fmt.Println(err)
		return JSONMessagesPage{}, err
	}

	retMsg := map[string]any{
		"payload": map[string]float64{
			"last_residual":  result.LastResidual,
			"last_predicted": result.LastPredicted,
			// "last_predicted": result,
		},
	}

	fmt.Printf("Reincarnate Hack: result = %+v\n", retMsg)

	return JSONMessagesPage{MessagesPage: MessagesPage{
		Total: 1,
		Messages: []Message{
			retMsg,
		},
	}}, err
}

// ARXMCAPredict applies the first-order ARX formula with fixed coefficients from the paper.
// Uses hourly data directly: Δ=4h maps to index offset 4, and 12h to offset 12.
// Returns the predicted MC_A at the last valid timestep.
func ARXMCAPredict(data []ARXDataPoint) (float64, error) {
	const minLen = 13
	if len(data) < minLen {
		return 0, errors.New("ARXMCAPredict: need at least 13 data points (index 12 requires t-12 for Temp)")
	}
	var pred float64
	for t := 12; t < len(data); t++ {
		pred = arxBeta0 +
			arxPhi1*data[t-4].SensorMCA +
			arxGamma1*data[t-4].AmbientRH +
			arxDelta3*data[t-12].AmbientT
	}
	return pred, nil
}

// arxRecord is an internal struct for the OLS workflow (time-sorted, numeric fields).
type arxRecord struct {
	Time time.Time
	RH   float64
	Temp float64
	MC   float64
}

// ARXOLSResult holds the outputs of the OLS-based ARX workflow.
type ARXOLSResult struct {
	// Coefficients: β0 (intercept), φ1 (MC lag 1), γ1 (RH lag 1), δ3 (Temp lag 3)
	Coefficients  []float64
	Predictions   []float64
	Residuals     []float64
	LastPredicted float64
	LastResidual  float64
}

// ARXMCAPredictOLS implements the full ARX workflow: sort, resample to 4h, build design matrix,
// run OLS regression, compute predictions and residuals. Returns coefficients, predictions,
// residuals, and the last predicted value and residual for convenience.
func ARXMCAPredictOLS(data []ARXDataPoint) (*ARXOLSResult, error) {
	if len(data) == 0 {
		return nil, errors.New("ARXMCAPredictOLS: empty dataset")
	}

	records, err := arxDataToRecords(data)
	if err != nil {
		return nil, err
	}

	sort.Slice(records, func(i, j int) bool {
		return records[i].Time.Before(records[j].Time)
	})

	resampled := arxResample4H(records)
	if len(resampled) < 4 {
		return nil, errors.New("ARXMCAPredictOLS: need at least 4 points after resampling to 4h grid")
	}

	X, Y := arxBuildMatrices(resampled)

	beta, err := arxOLS(X, Y)
	if err != nil {
		return nil, err
	}

	pred := arxPredict(X, beta)
	res := arxResiduals(Y, pred)

	coef := make([]float64, 4)
	for i := 0; i < 4; i++ {
		coef[i] = beta.AtVec(i)
	}

	n := len(pred)
	return &ARXOLSResult{
		Coefficients:  coef,
		Predictions:   pred,
		Residuals:     res,
		LastPredicted: pred[n-1],
		LastResidual:  res[n-1],
	}, nil
}

func arxDataToRecords(data []ARXDataPoint) ([]arxRecord, error) {
	records := make([]arxRecord, len(data))
	layout := "2006-01-02T15:04:05"
	for i, d := range data {
		t, err := time.Parse(layout, d.Time)
		if err != nil {
			t, err = time.Parse(time.RFC3339, d.Time)
			if err != nil {
				return nil, err
			}
		}
		records[i] = arxRecord{
			Time: t,
			RH:   d.AmbientRH,
			Temp: d.AmbientT,
			MC:   d.SensorMCA,
		}
	}
	return records, nil
}

func arxResample4H(data []arxRecord) []arxRecord {
	buckets := make(map[int64][]arxRecord)
	for _, r := range data {
		bt := r.Time.Truncate(4 * time.Hour).Unix()
		buckets[bt] = append(buckets[bt], r)
	}

	var result []arxRecord
	for _, values := range buckets {
		var rh, temp, mc float64
		for _, v := range values {
			rh += v.RH
			temp += v.Temp
			mc += v.MC
		}
		n := float64(len(values))
		result = append(result, arxRecord{
			Time: values[0].Time.Truncate(4 * time.Hour),
			RH:   rh / n,
			Temp: temp / n,
			MC:   mc / n,
		})
	}

	sort.Slice(result, func(i, j int) bool {
		return result[i].Time.Before(result[j].Time)
	})
	return result
}

func arxBuildMatrices(data []arxRecord) (*mat.Dense, *mat.VecDense) {
	rows := len(data) - 3
	X := mat.NewDense(rows, 4, nil)
	Y := mat.NewVecDense(rows, nil)

	for i := 3; i < len(data); i++ {
		r := i - 3
		X.Set(r, 0, 1.0)
		X.Set(r, 1, data[i-1].MC)
		X.Set(r, 2, data[i-1].RH)
		X.Set(r, 3, data[i-3].Temp)
		Y.SetVec(r, data[i].MC)
	}
	return X, Y
}

func arxOLS(X *mat.Dense, Y *mat.VecDense) (*mat.VecDense, error) {
	var xtx mat.Dense
	xtx.Mul(X.T(), X)

	var xty mat.VecDense
	xty.MulVec(X.T(), Y)

	var beta mat.VecDense
	if err := beta.SolveVec(&xtx, &xty); err != nil {
		return nil, err
	}
	return &beta, nil
}

func arxPredict(X *mat.Dense, beta *mat.VecDense) []float64 {
	rows, _ := X.Dims()
	pred := make([]float64, rows)
	for i := 0; i < rows; i++ {
		var val float64
		for j := 0; j < beta.Len(); j++ {
			val += X.At(i, j) * beta.AtVec(j)
		}
		pred[i] = val
	}
	return pred
}

func arxResiduals(y *mat.VecDense, pred []float64) []float64 {
	n := y.Len()
	res := make([]float64, n)
	for i := 0; i < n; i++ {
		res[i] = y.AtVec(i) - pred[i]
	}
	return res
}
