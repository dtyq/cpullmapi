package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/dtyq/cpullmapi"
)

func newServeCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "serve",
		Short: "Start the HTTP server",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runServe(cmd)
		},
	}
}

func runServe(cmd *cobra.Command) error {
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("failed to read config file: %w", err)
	}

	config, err := cpullmapi.ConfigFromYAML(configData)
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	if err := config.MiscInitialize(); err != nil {
		return fmt.Errorf("failed to initialize misc: %w", err)
	}
	defer config.MiscShutdown()

	server, err := config.CreateServer()
	if err != nil {
		return fmt.Errorf("failed to create server: %w", err)
	}

	return server.Run()
}
