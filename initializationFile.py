from libraries import os

def loadScreenFiles():
    csv_files = [f for f in os.listdir('data') if f.endswith('.csv')]
    print(csv_files)
    return csv_files

