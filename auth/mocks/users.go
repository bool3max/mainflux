package mocks

import (
	"context"
	"strings"

	protomfx "github.com/MainfluxLabs/mainflux/pkg/proto"
	"github.com/MainfluxLabs/mainflux/users"
	"google.golang.org/grpc"
)

var _ protomfx.UsersServiceClient = (*usersServiceClientMock)(nil)

type usersServiceClientMock struct {
	usersByID     map[string]users.User
	usersByEmails map[string]users.User
}

func NewUsersService(usersByID map[string]users.User, usersByEmails map[string]users.User) protomfx.UsersServiceClient {
	return &usersServiceClientMock{usersByID: usersByID, usersByEmails: usersByEmails}
}

func (svc *usersServiceClientMock) GetUsersByIDs(ctx context.Context, in *protomfx.UsersByIDsReq, opts ...grpc.CallOption) (*protomfx.UsersRes, error) {
	var users []*protomfx.User
	i := uint64(0)
	for _, id := range in.Ids {
		if user, ok := svc.usersByID[id]; ok {
			if in.Email != "" && !strings.Contains(user.Email, in.Email) {
				continue
			}

			if i >= in.Offset && i < in.Offset+in.Limit {
				users = append(users, &protomfx.User{Id: user.ID, Email: user.Email})
			}
			i++
		}
	}

	return &protomfx.UsersRes{Users: users, Limit: in.Limit, Offset: in.Offset, Total: i}, nil
}

func (svc *usersServiceClientMock) GetUsersByEmails(ctx context.Context, in *protomfx.UsersByEmailsReq, opts ...grpc.CallOption) (*protomfx.UsersRes, error) {
	var users []*protomfx.User
	for _, email := range in.Emails {
		if user, ok := svc.usersByEmails[email]; ok {
			users = append(users, &protomfx.User{Id: user.ID, Email: user.Email})
		}
	}

	return &protomfx.UsersRes{Users: users}, nil
}
