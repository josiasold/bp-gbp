import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='Tools for codes')
    parser.add_argument('-c','--code',default='steane',help='which code')
    parser.add_argument('-d','--dir',default='',help='output directory')

    args = parser.parse_args()

    code = args.code
    dir = args.dir

    if code == "steane":
        H_base = np.array([[1,0,0,1,0,1,1],
                           [0,1,0,1,1,0,1],
                           [0,0,1,0,1,1,1]],dtype=np.int32)

        n_c_x,n_q = H_base.shape
        n_c_z = n_c_x

        H_X = H_base.copy()
        H_Z = H_base.copy()

        H_gf2 = np.zeros((n_c_x+n_c_z,2*n_q),dtype=np.int32)
        H_gf2[:n_c_x,:n_q] = H_X
        H_gf2[n_c_x:,n_q:] = H_Z

        H_gf4 = np.zeros((n_c_x+n_c_z,n_q),dtype=np.int32)
        H_gf4[:n_c_x,:] = H_X
        H_gf4[n_c_x:,:] = 2*H_Z

        np.save(dir+'/'+code+'_hx.npy',H_X)
        np.save(dir+'/'+code+'_hz.npy',H_Z)
    
    elif code == "repetition":
        n_q = 5
        n_c = n_q-1
        H_base = np.eye(n_c,n_q,dtype=np.int32)
        for i in range(n_c):
            H_base[i,i+1] = 1
        
        n_c_x,__ = H_base.shape
        n_c_z = n_c_x


        H_X = H_base.copy()
        H_Z = H_base.copy()

        H_gf2 = np.zeros((n_c_x+n_c_z,2*n_q),dtype=np.int32)
        H_gf2[:n_c_x,:n_q] = H_X
        H_gf2[n_c_x:,n_q:] = H_Z

        H_gf4 = np.zeros((n_c_x+n_c_z,n_q),dtype=np.int32)
        H_gf4[:n_c_x,:] = H_X
        H_gf4[n_c_x:,:] = 2*H_Z

        np.save(dir+'/'+code+'_hx.npy',H_X)
        np.save(dir+'/'+code+'_hz.npy',H_Z)

    elif code == "repetition_loop":
        n_q = 5
        n_c = n_q-1
        H_base = np.eye(n_c,n_q,dtype=np.int32)
        for i in range(n_c):
            H_base[i,i+1] = 1
        
        H_base[2,4] = 1

        n_c_x,__ = H_base.shape
        n_c_z = n_c_x


        H_X = H_base.copy()
        H_Z = H_base.copy()

        H_gf2 = np.zeros((n_c_x+n_c_z,2*n_q),dtype=np.int32)
        H_gf2[:n_c_x,:n_q] = H_X
        H_gf2[n_c_x:,n_q:] = H_Z

        H_gf4 = np.zeros((n_c_x+n_c_z,n_q),dtype=np.int32)
        H_gf4[:n_c_x,:] = H_X
        H_gf4[n_c_x:,:] = 2*H_Z

        np.save(dir+'/'+code+'_hx.npy',H_X)
        np.save(dir+'/'+code+'_hz.npy',H_Z)



if __name__ == "__main__":
    main()