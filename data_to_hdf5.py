import tarfile
import numpy as np
import h5py
import time

start_time = time.time()

#   variables

# file names:
inname_list = ["compr3d/5mm/energy_scan.tgz"]
#inname_list = ["compr3d/5mm/angle_"+str(i)+".tgz" for i in [0,1,3,5,7,9]]
inname_list = ["compr3d/5mm/energy_scan.tgz"]
#inname_list = ["compr3d/5mm/angle_"+str(i)+".tgz" for i in [0,1,3,5,7,9]]

# for energy_scan:
num_dat_files = 18
num_dat_files = 18

# for angle_n files:
#num_dat_files = 6
#num_dat_files = 6

outname = "energy_data.h5"
outname = "energy_data.h5"


# Calorimeter structure
NL = 20
Nx = 110
Ny = 11
Ncell = NL*Nx*Ny
Nevt = 25000
cellevt = np.zeros((Nevt,NL,Nx,Ny),dtype=np.float32)
classical_energy = np.zeros(Nevt,dtype=np.float32)
scaling_factor = 11.49
scaling_factor = 11.49

i=0
#Adding shuffle
shuffled_indices = np.random.permutation(len(inname_list)*num_dat_files*Nevt)

with h5py.File(outname, 'w') as hf:
    data = hf.create_dataset('data', shape=(len(inname_list)*num_dat_files*Nevt,NL,Nx,Ny),
                            compression='gzip',chunks=(1, NL, Nx, Ny),dtype=np.float32)
    labels = hf.create_dataset('labels', shape=len (inname_list)*num_dat_files*Nevt,dtype=np.int16)
    energy = hf.create_dataset('energy',shape=(len(inname_list)*num_dat_files*Nevt),dtype=np.float32)
    for inname in inname_list:
        print(f"Opening {inname}")
        with tarfile.open(inname, "r:gz") as tar:
            # List all file names
            file_names = tar.getnames()
            # Get files with .dat extension
            dat_files = [name for name in file_names if name.endswith('.dat')]
            print(f"Detected {len(dat_files)} .dat files")
            if len(dat_files) != num_dat_files:
                print(f"Number of .dat files ({num_dat_files}) doesn't match detected number of files ({len(dat_files)})")
                exit()


            start_index = i*num_dat_files*Nevt
            for filename in dat_files:
                #reset cellevt
                cellevt.fill(0)
                classical_energy.fill(0)

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
                    classical_energy[ievt] = np.sum(cellevt[ievt])/scaling_factor
                    classical_energy[ievt] = np.sum(cellevt[ievt])/scaling_factor

                    Nread += Nlist
                
                label = np.int16(filename.split('_')[-1].split('.')[0])

                # this part below could potentially be sped, don't know why fancy indexing is
                # not working

                #Saving into hdf5 file
                end_index = start_index + Nevt
                for j in range(start_index,end_index):
                    data[shuffled_indices[j]] = cellevt[j-start_index]
                    labels[shuffled_indices[j]] = label
                    energy[shuffled_indices[j]] = classical_energy[j-start_index]

        
                # Update the start_index for the next iteration
                start_index = end_index
                
                print(Nread,"entries read from binary file ",inname+'/'+filename)
        i+=1

end_time = time.time()  # End the timer
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time:.2f} seconds")