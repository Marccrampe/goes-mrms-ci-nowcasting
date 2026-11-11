"""Verified Release download and actual-input/forecast visualisation for the demo."""
from pathlib import Path
import hashlib
import subprocess
import sys
import urllib.request

REPOSITORY='https://github.com/Marccrampe/goes-mrms-ci-nowcasting'
FILENAME='goes_mrms_ci35_hard_r2.pt'
EXPECTED='9231c9bb9aa540717986c1ae729cdbbf56629ed74b072de10a200be4f9ae507e'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data)
    return h.hexdigest()

def ensure_weights(root):
    path=Path(root)/'weights'/FILENAME;path.parent.mkdir(exist_ok=True)
    if path.exists():
        if sha(path)==EXPECTED:
            print('Released checkpoint SHA-256 verified.');return path
        with path.open('rb') as f:is_pointer=f.read(100).startswith(b'version https://git-lfs.github.com/spec/')
        if not is_pointer:raise RuntimeError('Existing checkpoint differs from the released SHA-256; inspect it before continuing.')
    url=REPOSITORY+'/releases/latest/download/'+FILENAME
    temp=path.with_suffix('.pt.download')
    print('Downloading the published Release checkpoint (309 MiB)...',flush=True)
    try:
        request=urllib.request.Request(url,headers={'User-Agent':'GOES-MRMS-research-demo'})
        with urllib.request.urlopen(request,timeout=60) as response,temp.open('wb') as out:
            total=0;last=0
            while True:
                data=response.read(4*1024*1024)
                if not data:break
                out.write(data);total+=len(data)
                if total>500*1024**2:raise RuntimeError('Unexpectedly large checkpoint download')
                if total-last>=32*1024**2:
                    print(f'{total/1024**2:.0f} MiB downloaded',flush=True);last=total
    except Exception as exc:
        raise RuntimeError('Checkpoint download failed. Verify that the repository and published Release '
            'are public and the asset is named '+FILENAME+'. You can also download it manually into weights/.') from exc
    if sha(temp)!=EXPECTED:raise RuntimeError('Downloaded asset SHA-256 mismatch; checkpoint was not installed.')
    temp.replace(path);print('Released checkpoint SHA-256 verified.')
    return path

def run_inference(root):
    root=Path(root).resolve();checkpoint=ensure_weights(root)
    output=root/'outputs/mini_case_output.npz'
    subprocess.run([sys.executable,str(root/'models/inference.py'),'--checkpoint',str(checkpoint),
        '--input',str(root/'examples/mini_case.npz'),'--normalization',str(root/'preprocessing/dataset/normalization.json'),
        '--output',str(output),'--expected-checkpoint-sha256',EXPECTED,'--device','cpu'],check=True)
    return output

def show_demo(root,window=0,history=3,display_max=.5):
    import matplotlib.pyplot as plt
    import numpy as np
    root=Path(root)
    with np.load(root/'examples/mini_case.npz') as data:
        goes=data['goes_bt_k'][history,5]
        radar=data['mrms_reflectivity_dbz'][history,0]
        valid=data['mrms_valid'][history,0]>0
    with np.load(root/'outputs/mini_case_output.npz') as data:probability=data['probability'][0,window]
    fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    offset=[-30,-20,-10,0][history];lead=['0–30 min','30–60 min'][window]
    for ax,title,field,cmap,lo,hi,unit in zip(axs,
        [f'GOES-East ABI C13 · t{offset:+d} min',f'MRMS · t{offset:+d} min',f'Raw probability · {lead}'],
        [goes,np.where(valid,radar,np.nan),probability],['gray_r','turbo','magma'],
        [210,0,0],[300,60,display_max],['Brightness temperature (K)','Reflectivity (dBZ)','Probability']):
        image=ax.imshow(field,cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest')
        ax.set_title(title,fontsize=10);ax.set_xticks([]);ax.set_yticks([])
        fig.colorbar(image,ax=ax,orientation='horizontal',shrink=.9,pad=.04,label=unit)
    plt.show()
    print(f'Forecast probability range: {probability.min():.6f}–{probability.max():.6f}')
    print('History selection and colour limits change the display only; the forecast remains fixed.')
