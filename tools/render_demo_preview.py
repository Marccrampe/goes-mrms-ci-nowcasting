"""Render actual miniature inputs and predictions; no synthetic forecast fields."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--repository',type=Path,required=True);a=p.parse_args()
root=a.repository.resolve()
with np.load(root/'examples/mini_case.npz') as z:
    goes=z['goes_bt_k'][-1,5]
    radar=z['mrms_reflectivity_dbz'][-1,0]
    valid=z['mrms_valid'][-1,0]>0
with np.load(root/'outputs/mini_case_output.npz') as z:prob=z['probability'][0]
fig,axes=plt.subplots(1,4,figsize=(13.2,3.6),layout='constrained')
for ax,title,field,cmap,low,high,unit in zip(axes,
    ['GOES ABI C13 · input','MRMS reflectivity · input','Forecast · 0–30 min','Forecast · 30–60 min'],
    [goes,np.where(valid,radar,np.nan),prob[0],prob[1]],
    ['gray_r','turbo','magma','magma'],[210,0,0,0],[300,60,1,1],['K','dBZ','Probability','Probability']):
    im=ax.imshow(field,origin='upper',cmap=cmap,vmin=low,vmax=high,interpolation='nearest')
    ax.set_title(title,fontsize=11,pad=9);ax.set_xticks([]);ax.set_yticks([])
    fig.colorbar(im,ax=ax,orientation='horizontal',shrink=.87,pad=.04,label=unit)
fig.suptitle('Real 64 × 64 TEST crop · released GOES–MRMS model',fontsize=13)
out=root/'docs/figures/demo_preview.png';out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out,dpi=170,facecolor='white');plt.close(fig)
print(out)
