//go:generate swag init -g swag.go -o docs -ot go

package cpullmapi

// @securityDefinitions.apikey Token
// @in header
// @name Authorization
// @description "Type 'Token TOKEN' to correctly set the API Key"
