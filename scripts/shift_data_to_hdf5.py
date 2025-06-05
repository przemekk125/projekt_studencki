import tarfile
import numpy as np
import h5py
import time

start_time = time.time()

#   variables

# file names:
inname_list = ["compr3d/5mm/angle_"+str(i)+".tgz" for i in [0,1,3,5,7,9]]
#inname_list = ["compr3d/5mm/angle_"+str(i)+".tgz" for i in [0,1,3,5,7,9]]

# for energy_scan:
#num_dat_files = 18


# for angle_n files:
num_dat_files = 6


outname = "angles_data3.h5"


# Calorimeter structure
NL = 20
Nx = 110
Ny = 11
Ncell = NL*Nx*Ny
Nevt = 25000
cellevt = np.zeros((Nevt,NL,Nx,Ny),dtype=np.float32)
classical_energy = np.zeros(Nevt,dtype=np.float32)
classical_shift = np.zeros((Nevt,4), dtype=np.float32)
shift_label = np.zeros((Nevt,4), dtype=np.float32)
scaling_factor = 11.49

i=0
#Adding shuffle
shuffled_indices = np.random.permutation(len(inname_list)*num_dat_files*Nevt)

with h5py.File(outname, 'w') as hf:
    data = hf.create_dataset('data', shape=(len(inname_list)*num_dat_files*Nevt,NL,Nx,Ny),
                            compression='gzip',chunks=(1, NL, Nx, Ny),dtype=np.float32)
    labels = hf.create_dataset('labels', shape=len (inname_list)*num_dat_files*Nevt,dtype=np.int16)
    shift_labels = hf.create_dataset('shift_labels', shape=(len(inname_list)*num_dat_files*Nevt,4), dtype=np.float32)
    energy = hf.create_dataset('energy',shape=(len(inname_list)*num_dat_files*Nevt),dtype=np.float32)
    shifts = hf.create_dataset('shifts', shape=(len(inname_list)*num_dat_files*Nevt,4), dtype=np.float32)
    
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

                    shift_label[ievt] = shift
                    
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

                    x = np.array([])
                    y = np.array([])
                    z = np.array([])
                    
                    # grubość warstwy (do ustalenia)
                    d=4

                    kalorymetr = cellevt[ievt,:,:,:]

                    for m in range(NL):
                        warstwa = kalorymetr[m,:,:]
                        rows, cols = np.indices(warstwa.shape)
                        total_mass = warstwa.sum()
                        if(total_mass!=0):
                            rows = rows*5 + 2.5
                            cols = cols*5 + 2.5
                            center_x = (rows * warstwa).sum() / total_mass
                            center_y = (cols * warstwa).sum() / total_mass
                            x = np.append(x,center_x)
                            y = np.append(y,center_y)
                            z = np.append(z,m*d+d/2)
                
                    ax, bx = np.polyfit(z, x, 1)
                    alphax = np.arctan(ax)
                    alphax = np.degrees(alphax)
                    x_pred = z*ax + bx
                    
                    ay, by = np.polyfit(z, y, 1)
                    alphay = np.arctan(ay)
                    alphay = np.degrees(alphay)
                    y_pred = ay*z + by

                    classical_shift[ievt,:] = np.array([bx, by, alphax, alphay])


                    Nread += Nlist
                
                label = np.int16(filename.split('_')[-1].split('.')[0])

                # this part below could potentially be sped, don't know why fancy indexing is
                # not working

                #Saving into hdf5 file
                end_index = start_index + Nevt

                print("start_index",start_index)
                print("end_index",end_index)
                for k in range(start_index,end_index):
                    data[shuffled_indices[k]] = cellevt[k-start_index]
                    labels[shuffled_indices[k]] = label
                    energy[shuffled_indices[k]] = classical_energy[k-start_index]
                    shifts[shuffled_indices[k]] = classical_shift[k-start_index]
                    shift_labels[shuffled_indices[k]] = shift_label[k-start_index]

        
                # Update the start_index for the next iteration
                start_index = end_index
                
                print(Nread ,"entries read from binary file ",inname+'/'+filename)
        i+=1

end_time = time.time()  # End the timer
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time:.2f} seconds")