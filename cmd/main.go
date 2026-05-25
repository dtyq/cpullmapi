package main

import (
	"flag"
	"log"
	"os"

	"github.com/dtyq/cpullmapi"
	_ "github.com/dtyq/cpullmapi/backends/crispasr"
	_ "github.com/dtyq/cpullmapi/backends/onnx"
	_ "github.com/dtyq/cpullmapi/backends/sherpa_onnx"
)

func main() {
	configPath := flag.String("config", "./config.yml", "path to config file")
	flag.Parse()

	configData, err := os.ReadFile(*configPath)
	if err != nil {
		log.Fatalf("failed to read config file: %v", err)
	}

	config, err := cpullmapi.ConfigFromYAML(configData)
	if err != nil {
		log.Fatalf("failed to load config: %v", err)
	}

	if err := config.MiscInitialize(); err != nil {
		log.Fatalf("failed to initialize misc: %v", err)
	}
	defer config.MiscShutdown()

	server, err := config.CreateServer()
	if err != nil {
		log.Fatalf("failed to create server: %v", err)
	}

	err = server.Run()
	if err != nil {
		log.Fatalf("failed to run server: %v", err)
	}
}
