import asyncio
import time

from pylsl import StreamInfo, StreamOutlet

## For Polar H10  sampling frequency ##
ECG_SAMPLING_FREQ = 130


async def send_timestamps():
    # Define the stream info
    info = StreamInfo('UnixTimestampStream', 'Timestamp', 1, ECG_SAMPLING_FREQ, 'double64',
                      'myuniquetimestampstream12345')

    # Create the stream outlet2
    outlet2 = StreamOutlet(info)

    print("Now sending timestamp data...")

    while True:
        # Get the current Unix timestamp in milliseconds
        timestamp = time.time() * 1000

        # Push the timestamp to the stream
        outlet2.push_sample([timestamp])

        # Sleep for a while before sending the next timestamp
        await asyncio.sleep(1 / ECG_SAMPLING_FREQ)


if __name__ == "__main__":
    # Run the asyncio event loop
    asyncio.run(send_timestamps())
