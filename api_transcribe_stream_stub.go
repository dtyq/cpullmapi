//go:build !with_audio

package cpullmapi

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

func (s *Server) transcribeStreamHandler(c *gin.Context) {
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

func (s *Server) transcribeRealtimeHandler(c *gin.Context) {
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
