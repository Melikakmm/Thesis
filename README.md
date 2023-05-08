# Estimating metallicity for RR Lyrae stars:

### Fourier Power Spectrum:
This method is used here to break down the light curve in order to understand the period of the light curve.
the idea is that, with Fourier transform the light curve is broken down into multiple curves with different frequencies. each of these decomposed curves have a specific power. Logically, the period of the light curve is deteremined and presented by the frequency of the most powerful curve.

A classical Fourier Transform requires the data to be evenly spaced in time, which is not likely to happen for astronomical observations.(due to weather, etc)
There is a type of Fourier ttansform algorithm used by astronomers, LOMB SCARGLE PERIODAGRAM, which is for unevenly spaced data in time.
