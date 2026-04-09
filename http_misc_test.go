package cpullmapi

import (
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"golang.org/x/crypto/argon2"
)

func TestTokenHash(t *testing.T) {
	_tokenBytes := make([]byte, 24)
	_, err := rand.Read(_tokenBytes)
	if !assert.NoError(t, err) {
		t.Fatalf("failed to read random bytes: %v", err)
	}

	tokenString := base64.StdEncoding.EncodeToString(_tokenBytes)
	tokenString = tokenString[:32]
	tokenBytes := []byte(tokenString)

	salt := make([]byte, 16)
	_, err = rand.Read(salt)
	if !assert.NoError(t, err) {
		t.Fatalf("failed to read random bytes: %v", err)
	}

	hash := argon2.IDKey(tokenBytes, salt, 1, 64*1024, 4, 32)
	if !assert.NoError(t, err) {
		t.Fatalf("failed to hash token: %v", err)
	}

	combined := append(salt, hash...)
	combinedBase64 := base64.StdEncoding.EncodeToString(combined)

	fmt.Println(tokenString)
	fmt.Println(combinedBase64)
}
