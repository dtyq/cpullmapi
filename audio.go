//go:build with_audio

package cpullmapi

import (
	"github.com/gopxl/beep/v2"
)

type DownMixMethod string

const (
	DownMixMethodAverage DownMixMethod = "average"
	DownMixMethodLeft    DownMixMethod = "left"
	DownMixMethodRight   DownMixMethod = "right"
)

type BeepDownMixer struct {
	source        beep.Streamer
	downMixMethod DownMixMethod
}

func NewBeepDownMixer(source beep.Streamer, method DownMixMethod) *BeepDownMixer {
	return &BeepDownMixer{
		source:        source,
		downMixMethod: method,
	}
}

func (d *BeepDownMixer) Stream(samples [][2]float64) (n int, ok bool) {
	n, ok = d.source.Stream(samples)
	if !ok || n == 0 {
		return
	}

	switch d.downMixMethod {
	case DownMixMethodAverage:
		for i := range n {
			samples[i][0] = (samples[i][0] + samples[i][1]) / 2
			samples[i][1] = samples[i][0]
		}
	case DownMixMethodLeft:
		for i := range n {
			samples[i][1] = samples[i][0]
		}
	case DownMixMethodRight:
		for i := range n {
			samples[i][0] = samples[i][1]
		}
	}
	return
}

func (d *BeepDownMixer) Err() error {
	return d.source.Err()
}

const beepConvertSamples = 4096

func BeepConvertSamplesToFloat32Array(streamer beep.Streamer) []float32 {
	length := 0
	bufs := [][][2]float64{}
	for {
		buf := make([][2]float64, beepConvertSamples)
		n, ok := streamer.Stream(buf)
		if !ok || n == 0 {
			break
		}
		length += n
		bufs = append(bufs, buf[:n])
	}
	ret := make([]float32, length)
	for i := range bufs {
		for j := range bufs[i] {
			if i*beepConvertSamples+j >= length {
				break
			}
			ret[i*beepConvertSamples+j] = float32(bufs[i][j][0])
		}
	}
	return ret
}
