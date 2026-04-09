package cpullmapi

import (
	"crypto/subtle"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"golang.org/x/crypto/argon2"
)

type ContextKeyCredentialOKType struct{}

var (
	ContextKeyCredentialOK = ContextKeyCredentialOKType{}
)

func (s *Server) checkCredential(c *gin.Context) {
	// get credential from TLS client certificate
	requestID := c.GetString(ContextKeyRequestID)
	if requestID == "" {
		requestID = "_"
	}
	logTag := requestID + "-authn"

	// http Authorization header authentication
	authHeader := c.GetHeader("Authorization")
	if authHeader == "" {
		// no credential found
		c.Next()
		return
	}

	parts := strings.SplitN(authHeader, " ", 2)
	if len(parts) != 2 {
		s.Logd(logTag, "invalid Authorization header: %s", authHeader)
		c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
			"code":    CodeUnauthorized,
			"message": "unauthorized",
		})
		return
	}
	scheme := parts[0]
	tokenString := parts[1]

	if scheme != "Token" {
		s.Logd(logTag, "invalid Authorization header: %s", authHeader)
		c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
			"code":    CodeUnauthorized,
			"message": "unauthorized",
		})
		return
	}

	// verify token
	salt := s.config.HTTP.TokenHash[:16]
	expectedTokenHash := s.config.HTTP.TokenHash[16:]
	providedTokenHash := argon2.IDKey([]byte(tokenString), salt, 1, 64*1024, 4, 32)
	if subtle.ConstantTimeCompare(providedTokenHash, expectedTokenHash) != 1 {
		c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
			"code":    CodeUnauthorized,
			"message": "invalid token",
		})
		return
	}

	c.Set(ContextKeyCredentialOK, true)
	c.Next()
}
