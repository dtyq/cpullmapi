package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"
)

var configPath string

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	root := newRootCommand()
	root.SilenceErrors = true
	root.SilenceUsage = true

	if err := root.ExecuteContext(ctx); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func newRootCommand() *cobra.Command {
	root := &cobra.Command{
		Use:   "cpullmapi",
		Short: "Run large-model inference on CPU",
		// 不带子命令时按 serve 走，跟以前一样。
		RunE: func(cmd *cobra.Command, args []string) error {
			return runServe(cmd)
		},
	}

	root.PersistentFlags().StringVar(&configPath, "config", "./config.yml", "path to the config file")

	root.AddCommand(newServeCommand())
	root.AddCommand(newDownloadCommand())

	return root
}
