import glob
import audioread

# wav_files = glob.glob('./audio/*.wav')
wav_files = glob.glob('./data/ATCO2-ASR/DATA/*.wav')
if len(wav_files) == 0:
    print('No Audio files found!')


# function to convert the information into 
# some readable format
def audio_duration(length):
    hours = length // 3600  # calculate in hours
    length %= 3600
    mins = length // 60  # calculate in minutes
    length %= 60
    seconds = length  # calculate in seconds
  
    return hours, mins, seconds  # returns the duration

length = 0
for file in wav_files:
    with audioread.audio_open(file) as f:
        totalsec = f.duration
        length = length+int(totalsec)
        print('Processed: {:.3f}%'.format(wav_files.index(file) / len(wav_files) * 100))


hours, mins, seconds = audio_duration(length)
print('Total Duration: {}:{}:{}'.format(hours, mins, seconds))
print('Audio Files   : {}'.format(len(wav_files)))

from datasets import load_dataset

atco2 = load_dataset('jlvdoorn/atco2-asr')
atcosim = load_dataset('jlvdoorn/atcosim')
atco2_atcosim = load_dataset('jlvdoorn/atco2-asr-atcosim')


def getDurInSec(sample):
    return len(sample['audio']['array'])/sample['audio']['sampling_rate']

def calcTotalDurInSec(dts):
    ttd_train = 0
    ttd_valid = 0
    for smp in dts['train']:
            ttd_train = ttd_train + getDurInSec(smp)
    for smp in dts['validation']:
            ttd_valid = ttd_valid + getDurInSec(smp)
            
    print('Dataset       : {}'.format(dts))
    print('Total Duration: {:.2f} Hours in {} files'.format((ttd_train+ttd_valid)/60/60, int(len(dts['train'])+len(dts['validation']))))
    print('Training      : {:.2f} Hours in {} files'.format(ttd_train/60/60, int(len(dts['train']))))
    print('Validation    : {:.2f} Hours in {} files'.format(ttd_valid/60/60, int(len(dts['validation']))))

print('ATCO2')
calcTotalDurInSec(atco2)
print('')
print('ATCOSIM')
calcTotalDurInSec(atcosim)
print('')
print('ATCO2-ATCOSIM')
calcTotalDurInSec(atco2_atcosim)