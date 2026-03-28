# Audio
SAMPLE_RATE: int = 44100
N_FFT: int = 2048
HOP_LENGTH: int = 512
N_MELS: int = 228

# Encoder
ENC_IDIM: int = 1253          # linear(1025) + mel(228)
ENC_HDIM: int = 512
ENC_ODIM: int = 24
ENC_NUM_LAYERS: int = 10
ENC_INTERMEDIATE_DIM: int = 2048
ENC_KERNEL_SIZE: int = 7

# Decoder
DEC_IDIM: int = 24
DEC_HDIM: int = 512
DEC_NUM_LAYERS: int = 10
DEC_INTERMEDIATE_DIM: int = 2048
DEC_KERNEL_SIZE: int = 7
DEC_HEAD_HDIM: int = 2048
DEC_HEAD_ODIM: int = 512      # × hop_length = waveform samples per frame

# Discriminators
MPD_PERIODS: tuple[int, ...] = (2, 3, 5, 7, 11)
MRD_FFT_SIZES: tuple[int, ...] = (512, 1024, 2048)

# Loss weights (SupertonicTTS §3.1)
LAMBDA_RECON: float = 45.0
LAMBDA_ADV: float = 1.0
LAMBDA_FM: float = 0.1

# Optimizer defaults
LEARNING_RATE: float = 2e-4
ADAM_BETAS: tuple[float, float] = (0.8, 0.99)
ADAM_EPS: float = 1e-9
LR_DECAY: float = 0.999
