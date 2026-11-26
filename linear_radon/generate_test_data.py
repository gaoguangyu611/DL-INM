"""Thirty independent holdout phantoms, no trajectory duplication or overwrites."""
from pathlib import Path
import hashlib,json
import numpy as np,h5py
from solver_runtime import get_operator,file_hash

def main():
    root=Path('cache')
    audit=json.loads((root/'split_audit.json').read_text())
    eligible=np.array(audit['eligible_source_indices'])
    assert len(eligible)>=30
    selection_seed=20261004
    noise_seed=20261005
    indices=np.sort(np.random.RandomState(selection_seed).choice(eligible,30,replace=False))
    rng=np.random.RandomState(noise_seed)
    R=get_operator()
    images=[]; observations=[]; rows=[]
    with h5py.File('data/randshepp.mat','r') as f:
        for j,idx in enumerate(indices):
            image=f['data'][:,:,int(idx)]
            image=(image/image.max()).astype(np.float32)
            gt=image.reshape(-1,1)
            exact=R @ gt
            std=0.02*np.max(np.abs(exact))
            y=(exact+std*rng.randn(*exact.shape)).astype(np.float32)
            noise=y.astype(np.float64)-exact
            images.append(gt); observations.append(y)
            rows.append({'sample':j,'source_index':int(idx),'delta':float(np.linalg.norm(noise)),
                'relative_noise':float(np.linalg.norm(noise)/np.linalg.norm(exact)),
                'gaussian_std_before_float32':float(std),
                'image_sha256':hashlib.sha256(gt.tobytes()).hexdigest()})
    images=np.array(images); observations=np.array(observations)
    assert len(set(r['image_sha256'] for r in rows))==30
    path=Path('data/test_dataset.npz')
    assert not path.exists(),'Existing 30-image dataset must be reviewed before replacement'
    np.savez_compressed(path,xk=images,ydelta=observations,
        xkdelta=np.zeros_like(images),sk=np.zeros_like(images),source_indices=indices)
    metadata={'description':'30 independent normalized random Shepp-Logan-type phantoms from randshepp.mat',
        'selection_seed':selection_seed,'noise_seed':noise_seed,'source_indices':indices.tolist(),
        'selection':'uniform without replacement from source images absent from all available train/validation NPZ xk fields',
        'source_pool':2500,'excluded_source_images':audit['excluded_source_count'],
        'eligible_source_images':len(eligible),'noise_model':'y_delta = R x + 0.02 max(abs(R x)) * standard Gaussian; stored float32',
        'image_shape':[128,128],'operator_shape':list(R.shape),'operator_dtype':str(R.dtype),
        'data_dtype':str(images.dtype),'dataset':str(path),'dataset_sha256':file_hash(path),
        'source_sha256':file_hash('data/randshepp.mat'),'operator_sha256':file_hash('R_limited.npy'),
        'placeholders':'xkdelta and sk are unused zero placeholders for original CustomDataset compatibility; no retraining',
        'samples':rows}
    (root/'dataset_metadata.json').write_text(json.dumps(metadata,indent=2))
    print('DATASET_CREATED',path,indices.tolist(),flush=True)


if __name__ == "__main__":
    main()
