// overshoot-publisher reads H.264 Annex B from stdin and publishes to a
// LiveKit room using the Go SDK's encoded frame API. Designed to be spawned
// by the Overshoot Python SDK as an optional high-performance transport.
//
// Usage:
//
//	ffmpeg -i <source> -c:v libx264 -f h264 pipe:1 | \
//	  overshoot-publisher --url wss://lk.example.com --token <jwt> --fps 6
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	livekit "github.com/livekit/protocol/livekit"
	lksdk "github.com/livekit/server-sdk-go/v2"
	"github.com/pion/webrtc/v4"
)

// statusWriter holds the original stderr fd for JSON status messages.
// Pion/ICE debug logging goes to os.Stderr which is redirected to /dev/null.
var statusWriter *os.File

func init() {
	// Duplicate the real stderr fd before redirecting os.Stderr to /dev/null.
	// This lets our JSON status messages reach the parent process while
	// pion/ICE debug noise (IPv6 route failures on macOS awdl0/llw0)
	// is silently discarded — preventing asyncio event loop starvation.
	fd, err := syscall.Dup(int(os.Stderr.Fd()))
	if err != nil {
		// Fallback: keep writing to stderr even if dup fails
		statusWriter = os.Stderr
		return
	}
	statusWriter = os.NewFile(uintptr(fd), "status")

	// Redirect os.Stderr to /dev/null so pion logs are discarded
	devNull, err := os.Open(os.DevNull)
	if err != nil {
		return
	}
	syscall.Dup2(int(devNull.Fd()), int(os.Stderr.Fd()))
	devNull.Close()
}

// statusMsg is written to the status writer as JSON for the parent Python process.
type statusMsg struct {
	Event   string `json:"event"`
	Message string `json:"message,omitempty"`
	Error   string `json:"error,omitempty"`
}

func emit(event, message, errStr string) {
	msg := statusMsg{Event: event, Message: message, Error: errStr}
	data, _ := json.Marshal(msg)
	fmt.Fprintln(statusWriter, string(data))
}

func main() {
	url := flag.String("url", "", "LiveKit server URL (required)")
	token := flag.String("token", "", "LiveKit JWT token (required)")
	fps := flag.Float64("fps", 6, "Frame rate of the H.264 stream")
	trackName := flag.String("track-name", "video", "Published track name")
	flag.Parse()

	if *url == "" || *token == "" {
		fmt.Fprintln(statusWriter, "error: --url and --token are required")
		os.Exit(1)
	}

	if err := run(*url, *token, *fps, *trackName); err != nil {
		emit("fatal", "", err.Error())
		os.Exit(1)
	}
}

func run(url, token string, fps float64, trackName string) error {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle SIGTERM/SIGINT for clean shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)
	go func() {
		<-sigCh
		emit("shutdown", "signal received", "")
		cancel()
	}()

	// Connect to LiveKit room
	roomCb := &lksdk.RoomCallback{
		ParticipantCallback: lksdk.ParticipantCallback{},
	}

	room, err := lksdk.ConnectToRoomWithToken(url, token, roomCb,
		lksdk.WithAutoSubscribe(false),
	)
	if err != nil {
		return fmt.Errorf("connect to room: %w", err)
	}
	defer room.Disconnect()

	emit("connected", fmt.Sprintf("room=%s", room.Name()), "")

	// Create track from stdin (H.264 Annex B)
	frameDuration := time.Duration(float64(time.Second) / fps)

	done := make(chan struct{})
	track, err := lksdk.NewLocalReaderTrack(os.Stdin, webrtc.MimeTypeH264,
		lksdk.ReaderTrackWithFrameDuration(frameDuration),
		lksdk.ReaderTrackWithOnWriteComplete(func() {
			emit("eof", "input stream ended", "")
			close(done)
		}),
	)
	if err != nil {
		return fmt.Errorf("create track: %w", err)
	}

	// Publish track with SOURCE_CAMERA to match Python SDK behavior
	pub, err := room.LocalParticipant.PublishTrack(track, &lksdk.TrackPublicationOptions{
		Name:   trackName,
		Source: livekit.TrackSource_CAMERA,
	})
	if err != nil {
		return fmt.Errorf("publish track: %w", err)
	}

	emit("publishing", fmt.Sprintf("track=%s fps=%.1f", pub.SID(), fps), "")

	// Wait for EOF or context cancellation
	select {
	case <-done:
		// Input ended (FFmpeg exited or pipe closed)
	case <-ctx.Done():
		// Signal received
	}

	return nil
}
