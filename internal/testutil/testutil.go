// Package testutil holds helpers shared by the test suites of the backends.
package testutil

import (
	"os"
	"testing"
)

// RequireFiles skips the calling test when any of the paths is missing, so
// tests that need local fixtures stay runnable on machines that lack them.
func RequireFiles(t testing.TB, paths ...string) {
	t.Helper()

	for _, path := range paths {
		if _, err := os.Stat(path); err != nil {
			t.Skipf("skipping: %s not found", path)
		}
	}
}
