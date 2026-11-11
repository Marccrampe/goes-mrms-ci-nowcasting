"""Original common-cohort loader; target transformation before crop selection."""
from ci_nowcasting.paths import activate_sources
activate_sources()
from ci_v4_training_support import CIGoesV4Dataset
from ci_target_inflation import transform


class TargetDataset(CIGoesV4Dataset):
    def __init__(self,*args,training_kind='hard',extra_pixels=2,dy_km,dx_km,**kwargs):
        self.training_kind=training_kind;self.extra_pixels=int(extra_pixels)
        self.dy_km=float(dy_km);self.dx_km=float(dx_km)
        super().__init__(*args,**kwargs)

    def _build_targets(self,rows):
        targets,masks,anchors=super()._build_targets(rows)
        if self.split=='train':
            targets=transform(targets,masks,self.extra_pixels,self.dy_km,self.dx_km,self.training_kind)
        return targets,masks,anchors
