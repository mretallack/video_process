#!/usr/bin/env python3
"""Video processor to concatenate two video files using PyAV library."""

import sys
import os
import av


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
        
        # Create output container with absolute path
        output = os.path.abspath(output)
        output_container = av.open(output, mode='w', format='webm')
        output_stream = output_container.add_stream('libvpx', rate=video_stream1.average_rate)
        output_stream.width = video_stream1.width
        output_stream.height = video_stream1.height
        output_stream.pix_fmt = 'yuv420p'
        output_stream.bit_rate = 5000000
        output_stream.options = {'crf': '4', 'b:v': '5M', 'quality': 'best', 'cpu-used': '0'}
        output_stream.time_base = video_stream1.time_base
        
        pts_offset = 0
        frame_duration = int(output_stream.time_base.denominator / output_stream.average_rate) * 2
        if logger:
            logger.info(f"Processing {input1} -> {output}")
        else:
            print(f"Processing {input1} -> {output}")
        
        # Process first video
        for packet in container1.demux(video_stream1):
            try:
                frames = video_stream1.decode(packet)
                for frame in frames:
                    # Convert frame to match output format
                    frame = frame.reformat(width=output_stream.width, height=output_stream.height, format=output_stream.pix_fmt)
                    frame.pts = pts_offset
                    frame.time_base = output_stream.time_base
                    pts_offset += frame_duration
                    
                    for out_packet in output_stream.encode(frame):
                        output_container.mux(out_packet)
            except (av.InvalidDataError, av.error.InvalidDataError):
                continue
        
        container1.close()
        
        # Process second video
        container2 = av.open(input2)
        video_stream2 = container2.streams.video[0]
        
        if logger:
            logger.info(f"Processing {input2} -> {output}")
        else:
            print(f"Processing {input2} -> {output}")
        
        for packet in container2.demux(video_stream2):
            try:
                frames = video_stream2.decode(packet)
                for frame in frames:
                    # Convert frame to match output format
                    frame = frame.reformat(width=output_stream.width, height=output_stream.height, format=output_stream.pix_fmt)
                    frame.pts = pts_offset
                    frame.time_base = output_stream.time_base
                    pts_offset += frame_duration
                    
                    for out_packet in output_stream.encode(frame):
                        output_container.mux(out_packet)
            except (av.InvalidDataError, av.error.InvalidDataError):
                continue
            except Exception as e:
                if logger:
                    logger.error(f"Error processing frame: {e}")
                else:
                    print(f"Error processing frame: {e}")
                continue

        # Flush encoder
        for out_packet in output_stream.encode():
            output_container.mux(out_packet)
        
        container2.close()
        output_container.close()
        
        if logger:
            logger.info(f"Video processing complete: {output}")
        else:
            print(f"Video processing complete: {output}")
        return 0
        
    except Exception as e:
        if logger:
            logger.error(f"Error: {e}")
        else:
            print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <input_video1> <input_video2> <output_video>")
        sys.exit(1)
    
    sys.exit(process_videos(sys.argv[1], sys.argv[2], sys.argv[3]))