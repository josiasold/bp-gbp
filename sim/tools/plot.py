import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse
from numpy.core.fromnumeric import size
import scipy as sc
import networkx as nx

def main():

    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 11
    rcParams['text.usetex'] = True

    parser = argparse.ArgumentParser(description='Tools to plot results of simulations')
    parser.add_argument('-t','--type',default='marg_mess',help='type of plots to show')
    parser.add_argument('-d','--dir',default='',help='output directory')
    parser.add_argument('-p','--plot',default='False',help='whether to plot')
    parser.add_argument('-i','--n_iter',default=0,help='number of iterations to plot')
    parser.add_argument('-f','--file_type',default='png',help='filetype')

    args = parser.parse_args()

    type = args.type
    dir = args.dir
    plot = args.plot
    n_iter = int(args.n_iter)
    file_type = args.file_type

    if type == 'tg':
        H = sc.sparse.csr_matrix(np.array([[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                                            [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                            [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
                                            [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
                                            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
                                            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                                            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                                            [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]]))
        graph = nx.algorithms.bipartite.matrix.from_biadjacency_matrix(H)
        top = nx.bipartite.sets(graph)[1]
        pos = nx.bipartite_layout(graph, top,aspect_ratio=10,scale=10,align='horizontal')
        labels = dict()
        v = 0
        c = 192
        for p in pos:
            if p < 12:
                labels[p] = c
                c+=1
            else:
                labels[p] = v
                v+=1
        nx.draw_networkx(graph,pos,labels = labels)
        plt.savefig(dir+'/testtg.'+ file_type)

    elif type == 'mess':
        data = np.genfromtxt('test.out',skip_header=1,delimiter='\t')
        edges = data[:,0]
        cq = data[:,1]
        qc = data[:,2]

        fig,ax = plt.subplots(1,1,constrained_layout=True)
        
        ax.plot(edges,cq,'ro',label='$c \to q$')
        ax.plot(edges,qc,'bo',label='$q \to c$')
        

        # ax.set_xlim(len(edges)/2,len(edges))
        ax.legend()

        fig.savefig('test.'+ file_type)
    
    elif type == 'marg_mess':
        marginals = np.load(dir+'/marginals.npy')
        messages = np.load(dir+'/messages.npy')
        hard_decisions = np.load(dir+'/hard_decisions.npy').astype(np.int32)
        # syndromes = np.load(dir+'/syndromes.npy').astype(np.int32)

        max_iter,n_qubits,n_paulis = marginals.shape

        if n_iter == 0:
            for i in range(1,max_iter):
                if marginals[i,0,0] == 1:
                    n_iter = i
                    break
            
        
        hd_1 = hard_decisions.copy()
        hd_1[hd_1!=0] = 1
        weights_hd = np.sum(hd_1,axis=1)
        # weights_s = np.sum(syndromes,axis=1)

        # fig,ax = plt.subplots(2,1,sharex=True,figsize=(25,25))
        # ax = ax.ravel()
        # ax[0].plot(range(max_iter),weights_hd,'o')
        # # ax[1].plot(range(max_iter),weights_s,'o')

        # ax[1].set_xlabel('iteration')
        # ax[0].set_ylabel('$|\hat{e}|$')
        # ax[0].grid()
        # ax[1].grid()
        # ax[1].set_ylabel('$|s|$')

        # fig.savefig(dir+'/hd_s.png')

        if plot == 'True':
            if n_iter == -1:
                n_iter = max_iter
            
            sel_qubits = [i for i in range(16)]
            square = np.int32(np.ceil(np.sqrt(len(sel_qubits))))
            
            fig,axes = plt.subplots(square,square,sharex=True,sharey=True,figsize=(25,25),constrained_layout=True)
            axes[-1,0].set_xlabel('iteration')
            axes[-1,0].set_ylabel('marginals')
            axes = axes.ravel()
            for iv,v in enumerate(sel_qubits):
                axes[iv].plot(marginals[1:n_iter,v,0],linestyle='dotted',color='k',marker='$I$')
                axes[iv].plot(marginals[1:n_iter,v,1],linestyle='dotted',color='b',marker='$X$',label='{}'.format(v))
                # axes[iv].plot(marginals[v,1:n_iter,2],linestyle='dotted',color='r',marker='$Z$')
                # axes[iv].plot(marginals[v,1:n_iter,3],linestyle='dotted',color='y',marker='$Y$')
                
                
                axes[iv].grid()
                axes[iv].legend()

                # axes[v].set_xlim(-0.01,15)#np.argmin(marginals[:,:,0]))
                axes[iv].set_ylim(-0.05,1.05)
                # axes[iv].set_ylim(0.4999999,0.5000001)
            
            fig.savefig(dir+'/marginals.'+ file_type)


        max_iter,n_edges,n_paulis = messages.shape

        
        if n_iter == 0:
            n_iter = max_iter

        # diff_m_cq = np.abs(np.diff(messages[:,41:n_iter,0],axis=1))
        # diff_m_qc = np.abs(np.diff(messages[:,41:n_iter,1],axis=1))

        # diff_m_cq_max = np.sum(diff_m_cq,axis=1)
        # diff_m_qc_max = np.sum(diff_m_qc,axis=1)

        # # fig,ax = plt.subplots(1,1)
        # # ax.hist(diff_m_cq_max)
        # # fig.savefig(dir+'/diffhist.png')

        # sel_edges_cq = np.argwhere(diff_m_cq_max > 3).flatten()
        # sel_edges_qc = np.argwhere(diff_m_qc_max > 3).flatten()
        # sel_edges = np.union1d(sel_edges_cq,sel_edges_qc)

        sel_edges = np.arange(n_edges)


        if plot == 'True':

            square = np.int32(np.ceil(np.sqrt(len(sel_edges))))
            
            fig,axes = plt.subplots(square,square,sharex=True,sharey=True,figsize=(25,25),constrained_layout=True)
            axes[-1,0].set_xlabel('iteration')
            axes[-1,0].set_ylabel('messages')
            axes = axes.ravel()
            for ie,e in enumerate(sel_edges):
                axes[ie].plot(messages[1:n_iter,e,0],linestyle='dotted',marker='$I$',color='k',label='{}'.format(e))
                axes[ie].plot(messages[1:n_iter,e,1],linestyle='dotted',marker='$X$',color='b')
                
                
                # axes[ie].plot(diff_m_cq[e,1:n_iter],color='k',marker='$cq$')
                # axes[ie].plot(diff_m_qc[e,1:n_iter],color='k',marker='$cq$',label='{}'.format(e))

                axes[ie].grid()
                axes[ie].legend()

                # axes[ie].set_xlim(-0.01,15)#np.argmin(marginals[:,:,0]))
                # axes[ie].set_ylim(-0.05,1.05)
            
            fig.savefig(dir+'/messages.'+ file_type)


    elif type == 'marg_mess_bp_4':
        marginals = np.load(dir+'/marginals_bp.npy')
        messages = np.load(dir+'/messages_bp.npy')
        hard_decisions = np.load(dir+'/hard_decisions_bp.npy').astype(np.int32)
        syndromes = np.load(dir+'/syndromes_bp.npy').astype(np.int32)
        # free_energy = np.load(dir+'/free_energy_bp.npy')

        n_qubits,max_iter,n_paulis = marginals.shape
        
        hd_1 = hard_decisions.copy()
        hd_1[hd_1!=0] = 1
        weights_hd = np.sum(hd_1,axis=1)
        weights_s = np.sum(syndromes,axis=1)

        # fig,ax = plt.subplots(1,1,sharex=True,figsize=(15,15))
        # ax.plot(range(max_iter),free_energy,'o')

        # ax.set_xlabel('iteration')
        # ax.set_ylabel('Free energy $U$')
        # ax.grid()
        

        # fig.savefig(dir+'/U_bp.png')

        fig,ax = plt.subplots(2,1,sharex=True,figsize=(15,15),constrained_layout=True)
        ax = ax.ravel()
        ax[0].plot(range(max_iter),weights_hd,'o')
        ax[1].plot(range(max_iter),weights_s,'o')

        ax[1].set_xlabel('iteration')
        ax[0].set_ylabel('$|\hat{e}|$')
        ax[0].grid()
        ax[1].grid()
        ax[1].set_ylabel('$|s|$')

        fig.savefig(dir+'/hd_s_bp.'+ file_type)

        if plot == 'True':
            if n_iter == 0:
                n_iter = max_iter
            
            sel_qubits = [i for i in range(16)]
            square = np.int32(np.ceil(np.sqrt(len(sel_qubits))))
            
            fig,axes = plt.subplots(square,square,sharex=True,sharey=True,figsize=(25,25),constrained_layout=True)
            axes[-1,0].set_xlabel('iteration')
            axes[-1,0].set_ylabel('marginals')
            axes = axes.ravel()
            for iv,v in enumerate(sel_qubits):
                axes[iv].plot(marginals[v,1:n_iter,0],linestyle='dotted',color='k',marker='$I$')
                axes[iv].plot(marginals[v,1:n_iter,1],linestyle='dotted',color='b',marker='$X$',label='{}'.format(v))
                axes[iv].plot(marginals[v,1:n_iter,2],linestyle='dotted',color='r',marker='$Z$')
                axes[iv].plot(marginals[v,1:n_iter,3],linestyle='dotted',color='y',marker='$Y$')
                
                
                axes[iv].grid()
                axes[iv].legend()

                # axes[v].set_xlim(-0.01,15)#np.argmin(marginals[:,:,0]))
                axes[iv].set_ylim(-0.05,1.05)
            
            fig.savefig(dir+'/marginals_bp.'+ file_type)


        n_edges,max_iter,n_paulis,_ = messages.shape
        print(messages.shape)
        
        if n_iter == 0:
            n_iter = max_iter

        diff_m_cq = np.abs(np.diff(messages[:,41:n_iter,0,0],axis=1))
        diff_m_qc = np.abs(np.diff(messages[:,41:n_iter,1,0],axis=1))

        diff_m_cq_max = np.sum(diff_m_cq,axis=1)
        diff_m_qc_max = np.sum(diff_m_qc,axis=1)

        fig,ax = plt.subplots(1,1,constrained_layout=True)
        ax.hist(diff_m_cq_max)
        fig.savefig(dir+'/diffhist_bp.'+ file_type)

        sel_edges_cq = np.argwhere(diff_m_cq_max > 3).flatten()
        sel_edges_qc = np.argwhere(diff_m_qc_max > 3).flatten()
        sel_edges = np.union1d(sel_edges_cq,sel_edges_qc)
        # i=9
        sel_edges = [2610,2617,2624,2631,2638,2645,2652,2659,2666,2673,2680,2687]


        if plot == 'True':

            square = np.int32(np.ceil(np.sqrt(len(sel_edges))))
            
            fig,axes = plt.subplots(square,square,sharex=True,sharey=True,figsize=(25,25),constrained_layout=True)
            axes[-1,0].set_xlabel('iteration')
            axes[-1,0].set_ylabel('messages')
            axes = axes.ravel()
            for ie,e in enumerate(sel_edges):
                axes[ie].plot(messages[e,:n_iter,0,0],linestyle='dotted',marker='$I$',color='k',label='{}'.format(e))
                axes[ie].plot(messages[e,:n_iter,0,1],linestyle='dotted',marker='$X$',color='b')
                axes[ie].plot(messages[e,:n_iter,0,2],linestyle='dotted',marker='$Z$',color='r')
                axes[ie].plot(messages[e,:n_iter,0,3],linestyle='dotted',marker='$Y$',color='y')
                
                axes[ie].plot(messages[e,:n_iter,1,0],linestyle='dashed',marker='$I$',color='k')
                axes[ie].plot(messages[e,:n_iter,1,1],linestyle='dashed',marker='$X$',color='b')
                axes[ie].plot(messages[e,:n_iter,1,2],linestyle='dashed',marker='$Z$',color='r')
                axes[ie].plot(messages[e,:n_iter,1,3],linestyle='dashed',marker='$Y$',color='y')
                
                # axes[ie].plot(diff_m_cq[e,1:n_iter],color='k',marker='$cq$')
                # axes[ie].plot(diff_m_qc[e,1:n_iter],color='k',marker='$cq$',label='{}'.format(e))

                axes[ie].grid()
                axes[ie].legend()

                # axes[ie].set_xlim(-0.01,15)#np.argmin(marginals[:,:,0]))
                # axes[ie].set_ylim(-0.05,1.05)
            
            fig.savefig(dir+'/messages_bp.'+ file_type)
    
    elif type == 'marg_mess_bp':
        marginals = np.load(dir+'/marginals_bp.npy')
        messages = np.load(dir+'/messages_bp.npy')
        hard_decisions = np.load(dir+'/hard_decisions_bp.npy').astype(np.int32)
        syndromes = np.load(dir+'/syndromes_bp.npy').astype(np.int32)
        # free_energy = np.load(dir+'/free_energy_bp.npy')
        n_qubits,max_iter,n_paulis = marginals.shape
        __,n_checks = syndromes.shape
        diff_marginals = np.sum(np.abs(np.diff(marginals[:,:,0],axis=1)),axis=1)
        diff_syndromes = np.sum(np.abs(np.diff(syndromes[:,:],axis=0)),axis=0)

        fig,ax = plt.subplots(1,1,sharex=True,figsize=(15,15),constrained_layout=True)
        ax.plot(range(n_qubits),diff_marginals,'o')

        ax.set_xlabel('qubit')
        ax.set_ylabel('np.sum(np.abs(np.diff(marginals[:,:,0],axis=1)),axis=1)')
        ax.grid()

        ax.set_xlim(-0.5,15.5)
        
        fig.savefig(dir+'/diff_marginals_bp.'+ file_type)

        fig,ax = plt.subplots(1,1,sharex=True,figsize=(15,15),constrained_layout=True)
        ax.plot(range(n_checks),diff_syndromes,'o')

        ax.set_xlabel('check')
        ax.set_ylabel('np.sum(np.abs(np.diff(syndromes[:,:],axis=0)),axis=0)')
        ax.grid()

        ax.set_xlim(191.5,203.5)
        
        fig.savefig(dir+'/diff_syndromes_bp.'+ file_type)

        # for i in range(max_iter):
        #     print(i,' : ',np.argwhere(syndromes[i] != 0).flatten()-192)


        
        
        hd_1 = hard_decisions.copy()
        hd_1[hd_1!=0] = 1
        weights_hd = np.sum(hd_1,axis=1)
        weights_s = np.sum(syndromes,axis=1)

        # fig,ax = plt.subplots(1,1,sharex=True,figsize=(15,15))
        # ax.plot(range(max_iter),free_energy,'o')

        # ax.set_xlabel('iteration')
        # ax.set_ylabel('Free energy $U$')
        # ax.grid()
        

        # fig.savefig(dir+'/U_bp.'+ file_type)

        fig,ax = plt.subplots(2,1,sharex=True,figsize=(15,15),constrained_layout=True)
        ax = ax.ravel()
        ax[0].plot(range(max_iter),weights_hd,'o')
        ax[1].plot(range(max_iter),weights_s,'o')

        ax[1].set_xlabel('iteration')
        ax[0].set_ylabel('$|\hat{e}|$')
        ax[0].grid()
        ax[1].grid()
        ax[1].set_ylabel('$|s|$')

        fig.savefig(dir+'/hd_s_bp.'+ file_type)

        if plot == 'True':
            if n_iter == 0:
                n_iter = max_iter
            
            sel_qubits = [i for i in range(16)]
            square = np.int32(np.ceil(np.sqrt(len(sel_qubits))))
            
            fig,axes = plt.subplots(square,square,sharex=True,sharey=True,figsize=(25,25),constrained_layout=True)
            axes[-1,0].set_xlabel('iteration')
            axes[-1,0].set_ylabel('marginals')
            axes = axes.ravel()
            for iv,v in enumerate(sel_qubits):
                axes[iv].plot(marginals[v,1:n_iter,0],linestyle='dotted',color='k',marker='$I$')
                axes[iv].plot(marginals[v,1:n_iter,1],linestyle='dotted',color='b',marker='$X$',label='{}'.format(v))
                axes[iv].plot(marginals[v,1:n_iter,2],linestyle='dotted',color='r',marker='$Z$')
                axes[iv].plot(marginals[v,1:n_iter,3],linestyle='dotted',color='y',marker='$Y$')
                
                
                axes[iv].grid()
                axes[iv].legend()

                # axes[v].set_xlim(-0.01,15)#np.argmin(marginals[:,:,0]))
                axes[iv].set_ylim(-0.05,1.05)
            
            fig.savefig(dir+'/marginals_bp.'+ file_type)


        n_edges,max_iter,n_paulis = messages.shape
        print(messages.shape)
        
        if n_iter == 0:
            n_iter = max_iter


        diff_m_cq = np.abs(np.diff(messages[:,41:n_iter,0],axis=1))
        diff_m_qc = np.abs(np.diff(messages[:,41:n_iter,1],axis=1))

        diff_m_cq_max = np.sum(diff_m_cq,axis=1)
        diff_m_qc_max = np.sum(diff_m_qc,axis=1)

        fig,ax = plt.subplots(1,1)
        ax.hist(diff_m_cq_max)
        fig.savefig(dir+'/diffhist_bp.'+ file_type)

        sel_edges_cq = np.argwhere(diff_m_cq_max > 3).flatten()
        sel_edges_qc = np.argwhere(diff_m_qc_max > 3).flatten()
        sel_edges = np.union1d(sel_edges_cq,sel_edges_qc)
        # i=9
        # sel_edges = [2610,2617,2624,2631,2638,2645,2652,2659,2666,2673,2680,2687]


        if plot == 'True':

            square = np.int32(np.ceil(np.sqrt(len(sel_edges))))
            
            fig,axes = plt.subplots(square,square,sharex=True,sharey=True,figsize=(25,25),constrained_layout=True)
            axes[-1,0].set_xlabel('iteration')
            axes[-1,0].set_ylabel('messages')
            axes = axes.ravel()
            for ie,e in enumerate(sel_edges):
                axes[ie].plot(messages[e,:n_iter,0],linestyle='dotted',marker='$I$',color='k',label='{}'.format(e))
                axes[ie].plot(messages[e,:n_iter,1],linestyle='dotted',marker='$X$',color='b')
                
                # axes[ie].plot(messages[e,:n_iter,1,0],linestyle='dashed',marker='$I$',color='k')
                # axes[ie].plot(messages[e,:n_iter,1,1],linestyle='dashed',marker='$X$',color='b')
                
                # axes[ie].plot(diff_m_cq[e,1:n_iter],color='k',marker='$cq$')
                # axes[ie].plot(diff_m_qc[e,1:n_iter],color='k',marker='$cq$',label='{}'.format(e))

                axes[ie].grid()
                axes[ie].legend()

                # axes[ie].set_xlim(-0.01,15)#np.argmin(marginals[:,:,0]))
                # axes[ie].set_ylim(-0.05,1.05)
            
            fig.savefig(dir+'/messages_bp.'+ file_type)


    

if __name__ == "__main__":
    main()