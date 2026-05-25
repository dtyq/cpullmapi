package cpullmapi

type Capability string

const (
	CapabilityImageSegmentation Capability = "image-seg"
	CapabilityOfflineASR        Capability = "offline-asr"
)

type Inferencer interface {
	GetCapabilities() []Capability

	Close()
}

var InferencerFactoryMap = map[string]any /*func(config [T any]) (Inferencer, error)*/ {}

type DummyInferencer struct {
}

func (i *DummyInferencer) GetCapabilities() []Capability {
	return []Capability{}
}

func (i *DummyInferencer) Close() {
}
