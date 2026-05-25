//go:build !with_image

package cpullmapi

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

func (s *Server) mattingHandler(c *gin.Context) {
	if !c.GetBool(ContextKeyCredentialOK) {
		c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
			"code":    CodeUnauthorized,
			"message": "unauthorized",
		})
		return
	}

	c.JSON(http.StatusNotImplemented, map[string]interface{}{
		"code":    CodeNotImplemented,
		"message": "not implemented",
	})
}
