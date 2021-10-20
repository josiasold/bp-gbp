import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import rc
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import argparse
from numpy.core.fromnumeric import size
import scipy as sc
import networkx as nx

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0],tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def main():

    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 11
    rcParams['text.usetex'] = True

    parser = argparse.ArgumentParser(description='Tools to plot results of simulations')
    parser.add_argument('-t','--type',default='marg_mess',help='type of plots to show')
    parser.add_argument('-c','--code_type',default='surface',help='codetype')
    parser.add_argument('-d','--dir',default='',help='output directory')
    parser.add_argument('-i','--n_iter',default=0,help='number of iterations to plot')
    parser.add_argument('-f','--file_type',default='png',help='filetype')
    parser.add_argument('-p','--plot',default='False',help='whether to plot')
    parser.add_argument('-ps','--plot_size',default='10x10',help='plot size in cm')
    parser.add_argument('-po','--plot_order',default='square',help='plot order, square, linear')



    args = parser.parse_args()

    type = args.type
    code_type = args.code_type
    dir = args.dir
    plot = args.plot
    n_iter = int(args.n_iter)
    file_type = args.file_type
    plot_size = args.plot_size

    if args.plot_size.split('x')[0] == 'fw':
        plot_size_w = 11.80737
        plot_size_h = float(args.plot_size.split('x')[1])
    elif args.plot_size.split('x')[0] == 'hw':
        plot_size_w = 11.80737 / 2.0
        plot_size_h = float(args.plot_size.split('x')[1])
    else:
        plot_size_w = float(args.plot_size.split('x')[0])
        plot_size_h = float(args.plot_size.split('x')[1])
    fs = cm2inch(plot_size_w,plot_size_h)
    plot_order = args.plot_order

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
    
    elif type == 'mm_gbp':
        for pauli in ['z','x']:
            for rep_split in range(2):
                marginals = np.load(dir+'/marginals_{}_{}.npy'.format(pauli,rep_split))
                messages = np.load(dir+'/messages_{}_{}.npy'.format(pauli,rep_split))
                hard_decisions = np.load(dir+'/hard_decisions_{}_{}.npy'.format(pauli,rep_split)).astype(np.int32)
                syndromes = np.load(dir+'/syndromes_{}_{}.npy'.format(pauli,rep_split)).astype(np.int32)
                free_energy = np.load(dir+'/free_energy_{}_{}.npy'.format(pauli,rep_split))

                max_iter,n_qubits,n_paulis = marginals.shape

                if n_iter == 0:
                    n_iter = max_iter
                    for i in range(1,max_iter):
                        if marginals[i,0,0] == 1:
                            n_iter = i
                            break
                elif n_iter == -1:
                    n_iter = max_iter
                    
                for i in range(10):
                    print(np.argwhere(hard_decisions[i] == 1).flatten())

                weights_hd = np.sum(hard_decisions,axis=1)
                weights_hd[weights_hd<=0] = -1
                weights_s = np.sum(syndromes,axis=1)
                weights_s[weights_s<0] = -1

                fig,ax = plt.subplots(2,1,sharex=True,figsize=fs,constrained_layout=True)
                ax = ax.ravel()
                ax[0].plot(range(1,max_iter),weights_hd[1:],'o')
                ax[1].plot(range(1,max_iter),weights_s[1:],'o')

                ax[1].set_xlabel('iteration')
                ax[0].set_ylabel('$|\hat{e}|$')
                ax[0].grid()
                ax[1].grid()
                ax[1].set_ylabel('$|s|$')

                fig.savefig(dir+'/hd_s_{}_{}_gbp.'.format(pauli,rep_split)+ file_type)

                print('.hd_s_gbp {} {} done.'.format(pauli,rep_split))

                fig,ax = plt.subplots(1,1,sharex=True,figsize=fs,constrained_layout=True)
                ax.plot(range(max_iter),free_energy,'o')
                
                ax.set_xlabel('iteration')
                ax.set_ylabel('$F$')
                ax.grid()


                fig.savefig(dir+'/F_{}_{}_gbp.'.format(pauli,rep_split)+ file_type)


                print('.F_gbp {} {} done.'.format(pauli,rep_split))
                
                sel_qubits = [i for i in range(16)]
                square = np.int32(np.ceil(np.sqrt(len(sel_qubits))))
                
                fig,axes = plt.subplots(square,square,sharex=True,sharey=True,figsize=fs,constrained_layout=True)

                axes[-1,0].set_xlabel('iteration')
                axes[-1,0].set_ylabel('belief')
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
                
                fig.savefig(dir+'/marginals_{}_{}_gbp.'.format(pauli,rep_split)+ file_type)
                print('.marginals_gbp {} {} done.'.format(pauli,rep_split))

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

                # sel_edges = np.array([26,14,1])
                sel_edges = np.array([30,20,2,12])

                edge_labels = {0:r'\(19 \to 23\)',2:r'\(18 \to 39\)',20:r'\(13 \to 17\)',12:r'\(15 \to 18\)',3:r'\(18 \to 23\)',16:r'\(14 \to 35\)',30:r'\(10 \to 35\)',13:r'\(14 \to 39\)',1:r'\(19 \to 40\)',26:r'\(11 \to 36\)',14:r'\(14 \to 18\)'}

                square = np.int32(np.ceil(np.sqrt(len(sel_edges))))
                
                fig,axes = plt.subplots(1,4,sharex=True,sharey=True,figsize=fs,constrained_layout=True)
                axes[0].set_xlabel('iteration')
                axes[0].set_ylabel('messages')
                axes = axes.ravel()

                # print('messages.shape = ',messages.shape)

                for ie,e in enumerate(sel_edges):
                    # for j in range(messages.shape[2]):
                    to_plot = np.array(messages[1:n_iter,e,:])
                    

                    axes[ie].plot(to_plot[:,0],linestyle='dotted',marker='x',markersize=4,color='k')
                    axes[ie].plot(to_plot[:,1],linestyle='dotted',marker='x',markersize=4,color='r')

                    axes[ie].set_title(edge_labels[e])
                    
                    # axes[ie].plot(messages[1:n_iter,e,0],linestyle='dotted',marker='x',color='k',label='{}'.format(e))
                    # axes[ie].plot(messages[1:n_iter,e,1],linestyle='dotted',marker='x',color='r')
                    # axes[ie].plot(messages[1:n_iter,e,1],linestyle='dotted',marker='$X$',color='b')
                    
                    # axes[ie].plot(diff_m_cq[e,1:n_iter],color='k',marker='$cq$')
                    # axes[ie].plot(diff_m_qc[e,1:n_iter],color='k',marker='$cq$',label='{}'.format(e))

                    axes[ie].grid()
                    # axes[ie].legend()

                        # axes[ie].set_xlim(-0.01,15)#np.argmin(marginals[:,:,0]))
                        # axes[ie].set_ylim(-0.05,1.05)
                    
                    fig.savefig(dir+'/messages_{}_{}_gbp.'.format(pauli,rep_split) + file_type)
                
                print('.messages_gbp {} {} done.'.format(pauli,rep_split))

    elif type == 'mm_bp2':

        colors = {'x':'b','z':'r'}
        
        for pauli in ['x','z']:
            print('** {} **'.format(pauli))

            marginals = np.load(dir+'/marginals_'+ pauli +'_bp.npy')
            m_qc = np.load(dir+'/m_qc_'+ pauli +'_bp.npy')
            m_cq = np.load(dir+'/m_cq_'+ pauli +'_bp.npy')
            hard_decisions = np.load(dir+'/hard_decisions_'+ pauli +'_bp.npy').astype(np.int32)
            syndromes = np.load(dir+'/syndromes_'+ pauli +'_bp.npy').astype(np.int32)
            free_energy = np.load(dir+'/free_energy_'+ pauli +'_bp.npy')

            n_qubits,max_iter,n_paulis = marginals.shape
            __,n_checks = syndromes.shape
            diff_marginals = np.sum(np.abs(np.diff(marginals[:,:,0],axis=1)),axis=1)
            diff_syndromes = np.sum(np.abs(np.diff(syndromes[:,:],axis=0)),axis=0)
            
            reset_iter = False
            if n_iter == 0:
                reset_iter = True
                for i in range(1,max_iter):
                    if np.all(marginals[:,i,:] == 1):
                        n_iter = i
                        break
                    n_iter = max_iter
            elif n_iter == -1:
                n_iter = max_iter
            
            print('n_iter = ',n_iter)

            fig,ax = plt.subplots(1,1,sharex=True,figsize=(15,15),constrained_layout=True)
            ax.plot(range(n_qubits),diff_marginals,'o')

            ax.set_xlabel('qubit')
            ax.set_ylabel('np.sum(np.abs(np.diff(marginals[:,:,0],axis=1)),axis=1)')
            ax.grid()

            ax.set_xlim(-0.5,15.5)
            
            fig.savefig(dir+'/diff_marginals_'+ pauli +'_bp.'+ file_type)

            fig,ax = plt.subplots(1,1,sharex=True,figsize=(15,15),constrained_layout=True)
            ax.plot(range(n_checks),diff_syndromes,'o')

            ax.set_xlabel('check')
            ax.set_ylabel('np.sum(np.abs(np.diff(syndromes[:,:],axis=0)),axis=0)')
            ax.grid()

            ax.set_xlim(191.5,203.5)
            
            fig.savefig(dir+'/diff_syndromes_'+ pauli +'_bp.'+ file_type)

            # for i in range(max_iter):
            #     print(i,' : ',np.argwhere(syndromes[i] != 0).flatten()-192)


            
            
            hd_1 = hard_decisions.copy()
            hd_1[hd_1!=0] = 1
            weights_hd = np.sum(hd_1,axis=1)
            weights_s = np.sum(syndromes,axis=1)

            fig,ax = plt.subplots(1,1,sharex=True,figsize=(15,15))
            # print(free_energy[:n_iter])
            ax.plot(range(n_iter),free_energy[:n_iter],'o--')

            ax.set_xlabel('iteration')
            ax.set_ylabel('Free energy $F$')
            ax.grid()
            
            fig.savefig(dir+'/F_'+ pauli +'_bp.'+ file_type)

            print(".free_energy done.")

            fig,ax = plt.subplots(2,1,sharex=True,figsize=(15,15),constrained_layout=True)
            ax = ax.ravel()
            ax[0].plot(range(max_iter),weights_hd,'o')
            ax[1].plot(range(max_iter),weights_s,'o')

            ax[1].set_xlabel('iteration')
            ax[0].set_ylabel('$|\hat{e}|$')
            ax[0].grid()
            ax[1].grid()
            ax[1].set_ylabel('$|s|$')

            fig.savefig(dir+'/hd_s_'+ pauli +'_bp.'+ file_type)

            
            sel_qubits = np.argwhere(marginals[:,n_iter-1,0] < 0.99).flatten()
            # print("sel_qubits = ",sel_qubits)
            if len(sel_qubits) == 1:
                other_qubits = np.array([sel_qubits-1,sel_qubits-2,sel_qubits+1])
                sel_qubits = np.concatenate((sel_qubits,other_qubits),axis=None)
            elif len(sel_qubits) == 0:
                sel_qubits = np.array([0,1,2,3])
            # print("sel_qubits = ",sel_qubits)
            if pauli == 'z':
                sel_qubits = np.array([77,89,129,243])
            if pauli == 'x':
                sel_qubits = np.array([36,68,84,311])
            # sel_qubits = np.arange(16)
            # sel_qubits = np.array([46,59,64,65,74,76,83,103,126,135,170,184])
            if code_type == 'rep':
                sel_qubits = np.arange(5)

            if plot_order == 'square':
                square = np.int32(np.ceil(np.sqrt(len(sel_qubits))))
                fig,axes = plt.subplots(square,square,sharex=True,sharey=True,figsize=fs,constrained_layout=True)
                axes[-1,0].set_xlabel('iteration')
                axes[-1,0].set_ylabel('belief')

            elif plot_order == 'linear':
                fig,axes = plt.subplots(1,len(sel_qubits),sharex=True,sharey=True,figsize=fs,constrained_layout=True)
            
                axes[0].set_xlabel('iteration')
                axes[0].set_ylabel('belief')
            axes = axes.ravel()
            for iv,v in enumerate(sel_qubits):
                # print(marginals)
                axes[iv].plot(marginals[v,:n_iter,0],linestyle='dotted',color='k', linewidth=1,markersize=3,marker='$I$')
                axes[iv].plot(marginals[v,:n_iter,1],linestyle='dotted',color=colors[pauli], linewidth=1,markersize=3,marker='${}$'.format(pauli),label='{}'.format(v))

                
                axes[iv].set_title('{}'.format(v))
                axes[iv].grid()
                # axes[iv].legend()

                # axes[v].set_xlim(-0.01,15)#np.argmin(marginals[:,:,0]))
                axes[iv].set_ylim(-0.05,1.05)
            
            fig.savefig(dir+'/marginals_'+ pauli +'_bp.'+ file_type)

            print(".marginals done.")

            n_edges,max_iter,n_paulis = m_cq.shape


            if n_iter == -1:
                n_iter = max_iter


            # diff_m_cq = np.abs(np.diff(messages[:,:n_iter,0],axis=1))
            # diff_m_qc = np.abs(np.diff(messages[:,:n_iter,1],axis=1))

            # diff_m_cq_max = np.sum(diff_m_cq,axis=1)
            # diff_m_qc_max = np.sum(diff_m_qc,axis=1)

            # fig,ax = plt.subplots(1,1)
            # ax.hist(diff_m_cq_max)
            # fig.savefig(dir+'/diffhist_'+ pauli +'_bp.'+ file_type)

            # sel_edges_cq = np.argwhere(diff_m_cq_max > 3).flatten()
            # sel_edges_qc = np.argwhere(diff_m_qc_max > 3).flatten()
            # sel_edges = np.union1d(sel_edges_cq,sel_edges_qc)
            # i=9
            # sel_edges = np.arange(n_edges)
            sel_edges = []
            n_edges_plot = len(sel_edges)


            if len(sel_edges)>0:
                if plot_order == 'square':
                    square = np.int32(np.ceil(np.sqrt(n_edges_plot)))
                    fig,axes = plt.subplots(square,square,sharex=True,sharey=True,figsize=fs,constrained_layout=True)
                    axes[-1,0].set_xlabel('iteration')
                    axes[-1,0].set_ylabel('messages')
                elif plot_order == 'linear':
                    fig,axes = plt.subplots(1,n_edges_plot,sharex=True,sharey=True,figsize=fs,constrained_layout=True)
                    axes[0].set_xlabel('iteration')
                    axes[0].set_ylabel('messages')
                
                
                # square = np.int32(np.ceil(np.sqrt(len(sel_edges))))
                
                # fig,axes = plt.subplots(square,square,sharex=True,sharey=True,figsize=fs,constrained_layout=True)
                
                axes = axes.ravel()
                for ie,e in enumerate(sel_edges):
                    m_cq_I, = axes[n_edges_plot-1-ie].plot(np.arange(1,n_iter),m_cq[e,1:n_iter,0],linestyle='dashed',marker='$I$',color='k')
                    m_cq_X, = axes[n_edges_plot-1-ie].plot(np.arange(1,n_iter),m_cq[e,1:n_iter,1],linestyle='dashed',marker='${}$'.format(pauli),color='b')

                    m_qc_I, = axes[n_edges_plot-1-ie].plot(np.arange(n_iter),m_qc[e,:n_iter,0],linestyle='solid',marker='$I$',color='grey')
                    m_qc_X, = axes[n_edges_plot-1-ie].plot(np.arange(n_iter),m_qc[e,:n_iter,1],linestyle='solid',marker='${}$'.format(pauli),color='g')
                    
                    # axes[ie].plot(messages[e,:n_iter,1,0],linestyle='dashed',marker='$I$',color='k')
                    # axes[ie].plot(messages[e,:n_iter,1,1],linestyle='dashed',marker='$X$',color='b')
                    
                    # axes[ie].plot(diff_m_cq[e,1:n_iter],color='k',marker='$cq$')
                    # axes[ie].plot(diff_m_qc[e,1:n_iter],color='k',marker='$cq$',label='{}'.format(e))

                    axes[n_edges_plot-1-ie].grid()
                    l = axes[n_edges_plot-1-ie].legend([(m_cq_I,m_cq_X),(m_qc_I,m_qc_X)],['{}: cq'.format(e),'qc'], numpoints=1,handler_map={tuple: HandlerTuple(ndivide=None)},bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',ncol=2, mode="expand", borderaxespad=0.)

                    # axes[ie].set_xlim(-0.01,15)#np.argmin(marginals[:,:,0]))
                    # axes[ie].set_ylim(-0.05,1.05)
                
                fig.savefig(dir+'/messages_'+ pauli +'_bp.'+ file_type)
                print(".messages done.")

                if (reset_iter == True):
                    n_iter = 0

    elif type == 'mm_bp4R':
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

        if n_iter == 0:
            n_iter = max_iter
        
        if code_type == 'surface':
            sel_qubits = np.arange(n_qubits)
            sel_qubits = [10,25,30]
        else:
            sel_qubits = [i for i in range(16)]
        square = np.int32(np.ceil(np.sqrt(len(sel_qubits))))
        
        fig,axes = plt.subplots(square,square,sharex=True,sharey=True,figsize=(25,25),constrained_layout=True)
        axes[-1,0].set_xlabel('iteration')
        axes[-1,0].set_ylabel('belief')
        axes = axes.ravel()
        for iv,v in enumerate(sel_qubits):
            axes[iv].plot(marginals[v,1:n_iter,0],linestyle='dotted',color='k',marker='$I$')
            axes[iv].plot(marginals[v,1:n_iter,1],linestyle='dotted',color='b',marker='$X$',label='{}'.format(v))
            axes[iv].plot(marginals[v,1:n_iter,2],linestyle='dotted',color='r',marker='$Z$')
            axes[iv].plot(marginals[v,1:n_iter,3],linestyle='dotted',color='y',marker='$Y$')
            
            
            axes[iv].grid()
            axes[iv].legend()

            # axes[v].set_xlim(-0.01,15)#np.argmin(marginals[:,:,0]))
            # axes[iv].set_ylim(-0.05,1.05)
        
        fig.savefig(dir+'/marginals_bp.'+ file_type)
        print('.marginals done.')

        n_edges,max_iter,n_paulis,_ = messages.shape
        
        if n_iter == 0:
            for i in range(1,max_iter):
                if marginals[0,i,0] == 1:
                    n_iter = i
                    break
                n_iter = max_iter
        elif n_iter == -1:
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
        if code_type == 'surface':
            sel_edges = [10,25,30]
        else:
            sel_edges = [2610,2617,2624,2631,2638,2645,2652,2659,2666,2673,2680,2687]

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
            # axes[ie].set_ylim(-1.05,1.05)
        
        fig.savefig(dir+'/messages_bp.'+ file_type)
        print('.messages done.')
    
    elif type == 'mm_bp4KL':

        marginals = np.load(dir+'/marginals_bp.npy')
        messages = np.load(dir+'/messages_bp.npy')
        hard_decisions = np.load(dir+'/hard_decisions_bp.npy').astype(np.int32)
        syndromes = np.load(dir+'/syndromes_bp.npy').astype(np.int32)
        free_energy = np.load(dir+'/free_energy_bp.npy')

        # print(messages)

        n_qubits,max_iter,n_paulis = marginals.shape
        __,n_checks = syndromes.shape
        # diff_marginals = np.sum(np.abs(np.diff(marginals[:,:,0],axis=1)),axis=1)
        # diff_syndromes = np.sum(np.abs(np.diff(syndromes[:,:],axis=0)),axis=0)

        if n_iter == 0:
            for i in range(1,max_iter):
                if np.all(marginals[:,i,:] == 1):
                    n_iter = i
                    break
                n_iter = max_iter
        elif n_iter == -1:
            n_iter = max_iter

        # fig,ax = plt.subplots(1,1,sharex=True,figsize=fs,constrained_layout=True)
        # ax.plot(range(n_qubits),diff_marginals,'o')

        # ax.set_xlabel('qubit')
        # ax.set_ylabel('np.sum(np.abs(np.diff(marginals[:,:,0],axis=1)),axis=1)')
        # ax.grid()

        # ax.set_xlim(-0.5,15.5)
        
        # fig.savefig(dir+'/diff_marginals_bp.'+ file_type)

        # fig,ax = plt.subplots(1,1,sharex=True,figsize=fs,constrained_layout=True)
        # ax.plot(range(n_checks),diff_syndromes,'o')

        # ax.set_xlabel('check')
        # ax.set_ylabel('np.sum(np.abs(np.diff(syndromes[:,:],axis=0)),axis=0)')
        # ax.grid()

        # ax.set_xlim(191.5,203.5)
        
        # fig.savefig(dir+'/diff_syndromes_bp.'+ file_type)

        # for i in range(max_iter):
        #     print(i,' : ',np.argwhere(syndromes[i] != 0).flatten()-192)


        
        
        # hd_1 = hard_decisions.copy()
        # hd_1[hd_1!=0] = 1
        # weights_hd = np.sum(hd_1,axis=1)
        # weights_s = np.sum(syndromes,axis=1)

        # fig,ax = plt.subplots(1,1,sharex=True,figsize=fs,constrained_layout=True)
        # # print(free_energy[:n_iter])
        # ax.plot(range(n_iter),free_energy[:n_iter],'o--')

        # ax.set_xlabel('iteration')
        # ax.set_ylabel('Free energy $F$')
        # ax.grid()
        
        # fig.savefig(dir+'/F_bp.'+ file_type)

        # print(".free_energy done.")

        # fig,ax = plt.subplots(2,1,sharex=True,figsize=fs,constrained_layout=True)
        # ax = ax.ravel()
        # ax[0].plot(range(max_iter),weights_hd,'o')
        # ax[1].plot(range(max_iter),weights_s,'o')

        # ax[1].set_xlabel('iteration')
        # ax[0].set_ylabel('$|\hat{e}|$')
        # ax[0].grid()
        # ax[1].grid()
        # ax[1].set_ylabel('$|s|$')

        # fig.savefig(dir+'/hd_s_bp.'+ file_type)

        
        if code_type == 'surface':
            # sel_qubits = np.arange(n_qubits)
            sel_qubits = [8,10,18,19]
        elif code_type == 'rep':
            sel_qubits = np.arange(5)
        else:
            sel_qubits = [i for i in range(16)]
        square = np.int32(np.ceil(np.sqrt(len(sel_qubits))))
        
        fig,axes = plt.subplots(2,2,sharex=True,sharey=True,figsize=fs,constrained_layout=True)
        axes[-1,0].set_xlabel('iteration')
        axes[-1,0].set_ylabel('belief')
        axes = axes.ravel()

        for iv,v in enumerate(sel_qubits):
            print(marginals[v])
            axes[iv].plot(marginals[v,:n_iter,0],linestyle='dotted',color='k',marker='$I$',lw=1,ms=1, label='{}'.format(v))
            axes[iv].plot(marginals[v,:n_iter,1],linestyle='dotted',color='b',marker='$X$',lw=1,ms=1)
            axes[iv].plot(marginals[v,:n_iter,2],linestyle='dotted',color='r',marker='$Z$',lw=1,ms=1)
            axes[iv].plot(marginals[v,:n_iter,3],linestyle='dotted',color='y',marker='$Y$',lw=1,ms=1)
            
            axes[iv].set_ylabel(r'$q_{{{}}}$'.format(v))
            
            axes[iv].grid()
            # axes[iv].legend()

            # axes[v].set_xlim(-0.01,15)#np.argmin(marginals[:,:,0]))
            # axes[iv].set_ylim(-0.05,1.05)
        
        fig.savefig(dir+'/marginals_bp.'+ file_type)


        print(".marginals done.")

        n_edges,max_iter,n_paulis = messages.shape

        
        if n_iter == -1:
            n_iter = max_iter


        # diff_m_cq = np.abs(np.diff(messages[:,:n_iter,0],axis=1))
        # diff_m_qc = np.abs(np.diff(messages[:,:n_iter,1],axis=1))

        # diff_m_cq_max = np.sum(diff_m_cq,axis=1)
        # diff_m_qc_max = np.sum(diff_m_qc,axis=1)

        # fig,ax = plt.subplots(1,1)
        # ax.hist(diff_m_cq_max)
        # fig.savefig(dir+'/diffhist_bp.'+ file_type)

        # sel_edges_cq = np.argwhere(diff_m_cq_max > 0.8).flatten()
        # sel_edges_qc = np.argwhere(diff_m_qc_max > 0.8).flatten()
        # sel_edges = np.union1d(sel_edges_cq,sel_edges_qc)
        # i=9
        # print('sel_edges = ',sel_edges)
        if code_type == 'surface':
            sel_edges = [51,54,52,37,38,35,33,48]
        else:
            sel_edges = []
            # sel_edges = [2610,2617,2624,2631,2638,2645,2652,2659,2666,2673,2680,2687]
            # sel_edges = [1263,1264,1265,1266,1270,1271,1272,1273,1277,1278,1279,1280,1284,1285,1286,1287,1291,1292,1293,1294,1298,1299,1300,1301,1305,1306,1307,1308,1312,1313,1314,1315,1319,1320,1321,1322,1326,1327,1328,1329,1333,1334,1335,1336,1340,1341,1342,1343]
        # sel_edges = [1452, 1453, 1454, 1455, 1564, 1565, 1566, 1567, 1676, 1677, 1678, 1679, 1788, 1789, 1790, 1791, 1900, 1901, 1902, 1903, 2012, 2013, 2014, 2015, 2124, 2125, 2126, 2127, 2236, 2237, 2238, 2239, 2348, 2349, 2350, 2351, 2460, 2461, 2462, 2463, 2572, 2573, 2574, 2575, 2684, 2685, 2686, 2687]

        # sel_edges = []
        


        if len(sel_edges)>0:

            square = np.int32(np.ceil(np.sqrt(len(sel_edges))))
            
            fig,axes = plt.subplots(square,square,sharex=True,sharey=True,figsize=fs,constrained_layout=True)
            axes[-1,0].set_xlabel('iteration')
            axes[-1,0].set_ylabel('messages')
            axes = axes.ravel()
            for ie,e in enumerate(sel_edges):
                axes[ie].plot(messages[e,:n_iter,0],linestyle='dotted',marker='$cq$',color='g',label='{}: cq'.format(e))
                axes[ie].plot(messages[e,:n_iter,1],linestyle='dotted',marker='$qc$',color='purple',label='qc')
                
                # axes[ie].plot(messages[e,:n_iter,1,0],linestyle='dashed',marker='$I$',color='k')
                # axes[ie].plot(messages[e,:n_iter,1,1],linestyle='dashed',marker='$X$',color='b')
                
                # axes[ie].plot(diff_m_cq[e,1:n_iter],color='k',marker='$cq$')
                # axes[ie].plot(diff_m_qc[e,1:n_iter],color='k',marker='$cq$',label='{}'.format(e))

                axes[ie].grid()
                axes[ie].set_title('{}'.format(e))
                # axes[ie].legend()
                # axes[ie].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',ncol=2, mode="expand", borderaxespad=0.)

                # axes[ie].set_xlim(-0.01,15)#np.argmin(marginals[:,:,0]))
                # axes[ie].set_ylim(-0.05,1.05)
            
            fig.savefig(dir+'/messages_bp.'+ file_type)
            print(".messages done.")


    elif type == 'exp':
        data = np.genfromtxt(dir+'/test.out',skip_header=1,delimiter='\t')

        fig,ax = plt.subplots(1,1,sharex=True,figsize=(15,15),constrained_layout=True)
        ax.scatter(data[:,0],data[:,1])

        ax.set_ylabel('dec')
        ax.set_xlabel('$|\Gamma(C)| / |C|$')
        ax.grid()

        # ax.set_xlim(-0.5,15.5)
        
        fig.savefig(dir+'/exp.'+ file_type)


if __name__ == "__main__":
    main()