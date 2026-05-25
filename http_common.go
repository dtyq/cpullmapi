package cpullmapi

import (
	"net/http"
)

type HTTPCode int

const (
	CodeSuccess             HTTPCode = http.StatusOK
	CodeNotFound            HTTPCode = http.StatusNotFound
	CodeUnauthorized        HTTPCode = http.StatusUnauthorized
	CodeForbidden           HTTPCode = http.StatusForbidden
	CodeInternalServerError HTTPCode = http.StatusInternalServerError
	CodeNotImplemented      HTTPCode = http.StatusNotImplemented
)

type CommonResponse struct {
	Code    HTTPCode `json:"code"`
	Message string   `json:"message"`
}
