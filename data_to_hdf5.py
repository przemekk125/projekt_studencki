import tarfile
import numpy as np
import h5py
import time

start_time = time.time()

inname = "projekt_studencki/compr3d/5mm/energy_scan.tgz"
outname = "cal_data.h5"

# Calorimeter structure
NL = 20
Nx = 110
Ny = 11
Ncell = NL*Nx*Ny
Nevt = 25000
cellevt = np.zeros((Nevt,NL,Nx,Ny),dtype=np.float32)

with tarfile.open(inname, "r:gz") as tar:
    # List all file names
    file_names = tar.getnames()
    # Get files with .dat extension
    dat_files = [name for name in file_names if name.endswith('.dat')]

    with h5py.File(outname, 'w') as hf:
        data = hf.create_dataset('data', shape=(len(dat_files)*Nevt,NL,Nx,Ny),
                          compression='gzip',chunks=(1, NL, Nx, Ny),dtype=np.float32)
        #hf.create_dataset('labels', data=labels)

        start_index = 0
        for filename in dat_files:
            #reset cellevt
            cellevt.fill(0)

            infile = tar.extractfile(filename)

            Nread = 0
            for ievt in range(Nevt):
                # Read 32 bytes (4 int64 values for header)
                head_raw = infile.read(4 * 8)  # Each int64 is 8 bytes
                if not head_raw:
                    break  # End of file
                head = np.frombuffer(head_raw, dtype=np.int64)
                
                # Read 32 bytes (4 float64 values for shift)
                shift_raw = infile.read(4 * 8)
                shift = np.frombuffer(shift_raw, dtype=np.float64)
                
                # Read the list of indices and energy values
                Nlist = head[3]
                idlist_raw = infile.read(Nlist * 8)  # Each int64 is 8 bytes
                idlist = np.frombuffer(idlist_raw, dtype=np.int64)
                
                elist_raw = infile.read(Nlist * 8)  # Each float64 is 8 bytes
                elist = np.frombuffer(elist_raw, dtype=np.float64).astype(np.float32)

                # Decode cell index
                idl = idlist//100000
                idx = (idlist%100000)//100
                idy = idlist%100

                cellevt[ievt,idl,idx,idy] = elist
                
                Nread += Nlist
            
            #Saving into hdf5 file
            end_index = start_index + Nevt
            data[start_index:end_index] = cellevt
    
            # Update the start_index for the next iteration
            start_index = end_index
            
            print(Nread,"entries read from binary file ",inname+'/'+filename)

end_time = time.time()  # End the timer
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time:.2f} seconds")