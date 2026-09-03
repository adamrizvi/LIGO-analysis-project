# To activate .venv environment execute these commands in the terminal:
#   1. "python -m venv .venv"
#   2. ".\.venv\Scripts\Activate.ps1"
#   3. "pip install -r freeze_requirements.txt"

# IF required packages have NOT installed after completing above:
#   To install the required packages run "python -m pip install numpy matplotlib scipy" after activating the .venv environment

import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.optimize import curve_fit

t, A, B = np.loadtxt("the_data.dat", usecols = [0, 1, 2], unpack = True)
#print(t[range(0, 4)])

A_ft = fftpack.fft(A)   # Initial fft
B_ft = fftpack.fft(B)

def pre_whitening(t, y, n):
    y = y - np.mean(y)
    freqs = fftpack.fftfreq(len(t), d=(t[1]-t[0]))

    def sine_fitting_func(t,f,p,a,offset):
         return a*np.sin(t*2*np.pi*f+p)+offset

    for i in range(n):
        ft = fftpack.fft(y)

        positive = freqs > 0
        freq_est = freqs[positive][np.argmax(np.abs(ft[positive]))]
        #freq_est = freqs[np.argmax(np.abs(ft))]    # This version allows both +/- freqs
        amp_est = np.max(np.abs(ft))/len(y)*2
        
        matrixA = np.vstack([np.sin(freq_est*2*np.pi*t),np.cos(freq_est*2*np.pi*t),np.ones(len(t))]).T
        alpha = np.dot(matrixA.T,matrixA)
        beta = np.dot(matrixA.T,y)
        matrixC = np.linalg.inv(alpha)
        afit = np.dot(matrixC,beta)
        phase_est = np.arctan2(afit[1],afit[0])

        popt, pcov = curve_fit(sine_fitting_func,t,y,p0=[freq_est,phase_est,amp_est,0.])
        #results = []
        #results.append(popt)
        y = y-sine_fitting_func(t,*popt)

    y_ft = fftpack.fft(y) # Recalculate Fourier transform

    print("Frequency:", popt[0])
    print("Phase:", popt[1])
    print("Amplitude:", popt[2])
    print("Offset:", popt[3])
    
    return y, y_ft#, results

def signal_filtering(t, y, y_ft, fc_high, fc_low):
    freqs = fftpack.fftfreq(len(t), d=(t[1]-t[0]))

    filt_high = 1 - np.exp(-(0.5*freqs/fc_high)**2) # High-pass filter
    filt_low = np.exp(-(0.5*freqs/fc_low)**2) # Low-pass filter

    ft_filtering = y_ft*filt_high*filt_low
    filtered = np.real(fftpack.ifft(ft_filtering))
    filtered_ft = fftpack.fft(filtered) # Recalculate Fourier transform

    return filtered, filtered_ft

def clipper(y, clip_high, clip_low):
    y = np.clip(y, clip_low, clip_high)
    y_ft = fftpack.fft(y)
    return y, y_ft

def export_data(t, y):
    results = np.array([t, y]).T
    np.savetxt("results.txt", results, delimiter=" ")

def quick_plotter(t, y, y1):
    plt.plot(t, y, "tab:blue", label="Before")
    plt.plot(t, y1, "tab:orange", label="After")
    plt.xlabel("Time [s]")
    plt.ylabel("Function [Units]")
    plt.legend()
    plt.show()

def new_qp(t, y, y_ft, y1, y1_ft):
    fig, axs = plt.subplots(2, 2, sharex='col', figsize=(8, 9))
    plt.suptitle("Quickplotter")
    axs[0, 0].plot(t, y, "tab:blue")
    axs[1, 0].plot(t, y1, "tab:blue")
    axs[1, 0].set_xlabel("Time [sec]")

    freqs = fftpack.fftfreq(len(t), d=(t[1]-t[0]))
    axs[0, 1].plot(freqs[:len(freqs)//2],np.abs(y_ft[:len(y_ft)//2]), "tab:blue")
    axs[1, 1].plot(freqs[:len(freqs)//2],np.abs(y1_ft[:len(y1_ft)//2]), "tab:blue")
    axs[1, 1].set_xlabel("Frequency [Hz]")

    axs[0, 0].set_title("Signal", pad=25)
    axs[0, 1].set_title("Fourier Transform", pad=25)

    for ax in axs.flat:
        ax.set_ylabel("")
        fig.text(0.04, 0.5, "Relative Flux", va="center", rotation="vertical")

    plt.show()

A1, A1_ft = pre_whitening(t, A, 10)

A2, A2_ft = signal_filtering(t, A1, A1_ft, 10, 500)

A3, A3_ft = pre_whitening(t, A2, 10)

new_qp(t, A, A_ft, A3, A3_ft)

#A4, A4_ft = signal_filtering(t, A3, A3_ft, 50, 50)

#A5, A5_ft = clipper(A4, 15, -15)

#new_qp(t, A4, A4_ft, A5, A5_ft)

#export_data(t, A4)


"""
def pre_whitening(t, y):
    y = y - np.mean(y)
    ft = fftpack.fft(y)
    freqs = fftpack.fftfreq(len(t), d=(t[1]-t[0]))

    #positive = freqs > 0
    #freq_est = freqs[positive][np.argmax(np.abs(ft[positive]))]
    freq_est = freqs[np.argmax(np.abs(ft))]    # This version allows both +/- freqs
    amp_est = np.max(np.abs(ft))/len(y)*2

    matrixA = np.vstack([np.sin(freq_est*2*np.pi*t),np.cos(freq_est*2*np.pi*t),np.ones(len(t))]).T
    alpha = np.dot(matrixA.T,matrixA)
    beta = np.dot(matrixA.T,y)

    matrixC = np.linalg.inv(alpha)
    afit = np.dot(matrixC,beta)
    phase_est = np.arctan2(afit[1],afit[0])

    def sine_fitting_func(t,f,p,a,offset):
        return a*np.sin(t*2*np.pi*f+p)+offset

    popt, pcov = curve_fit(sine_fitting_func,t,y,p0=[freq_est,phase_est,amp_est,0.])
    print("Frequency:", popt[0])
    print("Phase:", popt[1])
    print("Amplitude:", popt[2])
    print("Offset:", popt[3])
    y = y-sine_fitting_func(t,*popt)
    return y

# print(np.max(np.abs(A1 - A2)))
"""