package cpullmapi

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

// @BasePath /api/v1

type HealthcheckResponse struct {
	Code       HTTPCode `json:"code" example:"200"`
	Message    string   `json:"message" example:"ok"`
	Authorized bool     `json:"authorized" example:"true"`
}

// @Summary healthcheck
// @Description do health check, return 200 if the server is healthy, otherwise return 500
// @Produce json
// @Success 200 {object} HealthcheckResponse "ok"
// @Failure 500 "internal server error"
// @Router /healthcheck [get]
func (s *Server) healthcheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"code":       CodeSuccess,
		"message":    "ok",
		"authorized": c.GetBool(ContextKeyCredentialOK),
	})
}
