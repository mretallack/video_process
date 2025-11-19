#!/usr/bin/env python3
"""Video processor to concatenate two video files using PyAV library."""

import sys
import os
import av
import traceback


def process_videos(input1, input2, output, logger=None):
    """Process two video files by concatenating their frames."""
    try:
        # Check if input files exist
        if not os.path.exists(input1):
            raise FileNotFoundError(f"Input file not found: {input1}")
        if not os.path.exists(input2):
            raise FileNotFoundError(f"Input file not found: {input2}")
        
        # Open first input video to get properties
        container1 = av.open(input1)
        video_stream1 = container1.streams.video[0]
        audio_stream1 = container1.streams.audio[0] if container1.streams.audio else None
        
        # Create output container with absolute path
        output = os.path.abspath(output)
        output_container = av.open(output, mode='w', format='webm')
        
        # Use fallback frame rate if average_rate is None
        frame_rate = video_stream1.average_rate or av.Rational(25, 1)
        
        output_video_stream = output_container.add_stream('libvpx', rate=frame_rate)
        output_video_stream.width = video_stream1.width
        output_video_stream.height = video_stream1.height
        output_video_stream.pix_fmt = 'yuv420p'
        output_video_stream.bit_rate = 5000000
        output_video_stream.options = {'crf': '4', 'b:v': '5M', 'quality': 'best', 'cpu-used': '0'}
        output_video_stream.time_base = video_stream1.time_base or av.Rational(1, 25)
        
        # Add audio stream if available
        output_audio_stream = None
        if audio_stream1:
            output_audio_stream = output_container.add_stream('libvorbis')
            output_audio_stream.rate = audio_stream1.rate
            output_audio_stream.format = audio_stream1.format
            output_audio_stream.layout = audio_stream1.layout
            output_audio_stream.time_base = audio_stream1.time_base or av.Rational(1, audio_stream1.rate)
        
        video_pts_offset = 0
        audio_pts_offset = 0
        frame_duration = int(output_video_stream.time_base.denominator / frame_rate) * 2
        if logger:
            logger.info(f"Processing {input1} -> {output}")
        else:
            print(f"Processing {input1} -> {output}")
        
        # Process first video and audio
        for packet in container1.demux():
            try:
                if packet.stream == video_stream1:
                    frames = video_stream1.decode(packet)
                    for frame in frames:
                        frame = frame.reformat(width=output_video_stream.width, height=output_video_stream.height, format=output_video_stream.pix_fmt)
                        frame.pts = video_pts_offset
                        frame.time_base = output_video_stream.time_base
                        video_pts_offset += frame_duration
                        
                        for out_packet in output_video_stream.encode(frame):
                            output_container.mux(out_packet)
                elif packet.stream == audio_stream1 and output_audio_stream:
                    frames = audio_stream1.decode(packet)
                    for frame in frames:
                        frame.pts = audio_pts_offset
                        frame.time_base = output_audio_stream.time_base
                        audio_pts_offset += frame.samples
                        
                        for out_packet in output_audio_stream.encode(frame):
                            output_container.mux(out_packet)
            except (av.InvalidDataError, av.error.InvalidDataError) as e:
                if logger:
                    logger.debug(f"Invalid data in video1 packet: {e}")
                continue
            except Exception as e:
                if logger:
                    logger.error(f"Error processing video1 packet: {e}\n{traceback.format_exc()}")
                else:
                    print(f"Error processing video1 packet: {e}\n{traceback.format_exc()}")
                continue
        
        container1.close()
        
        # Process second video
        container2 = av.open(input2)
        video_stream2 = container2.streams.video[0]
        audio_stream2 = container2.streams.audio[0] if container2.streams.audio else None
        
        if logger:
            logger.info(f"Processing {input2} -> {output}")
        else:
            print(f"Processing {input2} -> {output}")
        
        for packet in container2.demux():
            try:
                if packet.stream == video_stream2:
                    frames = video_stream2.decode(packet)
                    for frame in frames:
                        frame = frame.reformat(width=output_video_stream.width, height=output_video_stream.height, format=output_video_stream.pix_fmt)
                        frame.pts = video_pts_offset
                        frame.time_base = output_video_stream.time_base
                        video_pts_offset += frame_duration
                        
                        for out_packet in output_video_stream.encode(frame):
                            output_container.mux(out_packet)
                elif packet.stream == audio_stream2 and output_audio_stream:
                    frames = audio_stream2.decode(packet)
                    for frame in frames:
                        frame.pts = audio_pts_offset
                        frame.time_base = output_audio_stream.time_base
                        audio_pts_offset += frame.samples
                        
                        for out_packet in output_audio_stream.encode(frame):
                            output_container.mux(out_packet)
            except (av.InvalidDataError, av.error.InvalidDataError) as e:
                if logger:
                    logger.debug(f"Invalid data in video2 packet: {e}")
                continue
            except Exception as e:
                if logger:
                    logger.error(f"Error processing video2 packet: {e}\n{traceback.format_exc()}")
                else:
                    print(f"Error processing video2 packet: {e}\n{traceback.format_exc()}")
                continue

        # Flush encoders
        try:
            for out_packet in output_video_stream.encode():
                output_container.mux(out_packet)
        except Exception as e:
            if logger:
                logger.error(f"Error flushing video encoder: {e}\n{traceback.format_exc()}")
            else:
                print(f"Error flushing video encoder: {e}\n{traceback.format_exc()}")
        
        if output_audio_stream:
            try:
                for out_packet in output_audio_stream.encode():
                    output_container.mux(out_packet)
            except Exception as e:
                if logger:
                    logger.error(f"Error flushing audio encoder: {e}\n{traceback.format_exc()}")
                else:
                    print(f"Error flushing audio encoder: {e}\n{traceback.format_exc()}")
        
        container2.close()
        output_container.close()
        
        if logger:
            logger.info(f"Video processing complete: {output}")
        else:
            print(f"Video processing complete: {output}")
        return 0
        
    except Exception as e:
        if logger:
            logger.error(f"Fatal error in process_videos: {e}\n{traceback.format_exc()}")
        else:
            print(f"Fatal error in process_videos: {e}\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <input_video1> <input_video2> <output_video>")
        sys.exit(1)
    
    sys.exit(process_videos(sys.argv[1], sys.argv[2], sys.argv[3]))