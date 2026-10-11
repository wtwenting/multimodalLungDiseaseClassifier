"""Run Coswara B1/B2/B3 imbalance experiments across 2D CNN, 1D CNN and LSTM."""
from pathlib import Path
import argparse, csv, json, time, zlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
import librosa
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from psupr_proj.features import preprocess_coswara_waveform, create_log_mel
from psupr_proj.models import build_baseline_2d_cnn, build_baseline_1d_cnn, build_baseline_lstm

CLASSES=["healthy","COVID-19"]; C2I={x:i for i,x in enumerate(CLASSES)}
SR=16000; DUR=10.; NS=160000; TOP_DB=60.; FFT=1024; HOP=256; NMELS=64; NMFCC=13
SHAPES={"logmel":(64,626,1),"waveform":(160000,1),"mfcc":(626,13)}
REPS={"2d_cnn":"logmel","1d_cnn":"waveform","lstm":"mfcc"}
EXPS={
"B1_class_weight":{"cw":True,"balance":False,"aug":False},
"B2_covid_aug":{"cw":False,"balance":False,"aug":True},
"B3_balanced_aug":{"cw":False,"balance":True,"aug":True},
}

def records(meta, split):
    with Path(meta).open(newline="",encoding="utf-8") as f:
        rows=[r for r in csv.DictReader(f) if r["split"].strip()==split and r["status"].strip() in C2I]
    if not rows: raise ValueError(f"No {split} records")
    return rows

def augment(x, seed):
    rng=np.random.default_rng(seed); x=x.astype(np.float32,copy=True)
    x*=rng.uniform(.9,1.1)
    x+=rng.normal(0,max(float(np.max(np.abs(x)))*.005,1e-5),x.shape).astype(np.float32)
    m=max(1,int(.1*len(x))); s=int(rng.integers(-m,m+1)); y=np.zeros_like(x)
    if s>0:y[s:]=x[:-s]
    elif s<0:y[:s]=x[-s:]
    else:y=x
    return y

def represent(path, rep, do_aug, seed):
    x,_=preprocess_coswara_waveform(Path(path),sample_rate=SR,target_duration=DUR,top_db=TOP_DB)
    if do_aug:x=augment(x,seed)
    if rep=="waveform":return x[:,None].astype(np.float32)
    if rep=="logmel":return create_log_mel(x,SR,n_mels=NMELS,n_fft=FFT,hop_length=HOP)[...,None].astype(np.float32)
    return librosa.feature.mfcc(y=x,sr=SR,n_mfcc=NMFCC,n_fft=FFT,hop_length=HOP).T.astype(np.float32)

def dataset(meta,split,rep,batch,shuffle,aug=False,balance=False,seed=42):
    rs=records(meta,split)
    if balance:
        if split!="train":raise ValueError("Balance train only")
        h=[r for r in rs if r["status"].strip()=="healthy"]; c=[r for r in rs if r["status"].strip()=="COVID-19"]
        rng=np.random.default_rng(seed)
        rs=rs+[dict(c[int(i)]) for i in rng.integers(0,len(c),size=max(0,len(h)-len(c)))]
    paths=[r["audio_path"] for r in rs]; labels=[C2I[r["status"].strip()] for r in rs]
    ds=tf.data.Dataset.from_tensor_slices((paths,labels))
    if shuffle:ds=ds.shuffle(len(rs),seed=seed,reshuffle_each_iteration=True)
    counter=iter(range(10**9))
    def proc(p,y):
        def py(pp,yy):
            ps=pp.numpy().decode(); lab=int(yy.numpy())
            do=split=="train" and aug and lab==1
            # presentation counter ensures repeated/epoch presentations can vary reproducibly within run
            sd=seed+zlib.crc32(ps.encode())+next(counter)
            return represent(ps,rep,do,sd),np.int64(lab)
        x,z=tf.py_function(py,[p,y],[tf.float32,tf.int64]); x.set_shape(SHAPES[rep]); z.set_shape(())
        return x,z
    return ds.map(proc,num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE),len(rs)

def build(key):
    if key=="2d_cnn":return build_baseline_2d_cnn(input_shape=SHAPES["logmel"],num_classes=2)
    if key=="1d_cnn":return build_baseline_1d_cnn(input_shape=SHAPES["waveform"],num_classes=2)
    return build_baseline_lstm(input_shape=SHAPES["mfcc"],num_classes=2)

def weights(meta):
    rs=records(meta,"train"); n=len(rs); counts={c:sum(r["status"].strip()==c for r in rs) for c in CLASSES}
    return {C2I[c]:n/(2*counts[c]) for c in CLASSES}

def evaluate(model,ds):
    yt=[];yp=[]
    for x,y in ds:
        yt+=y.numpy().tolist(); yp+=np.argmax(model.predict(x,verbose=0),axis=1).tolist()
    yt=np.array(yt);yp=np.array(yp); rep=classification_report(yt,yp,labels=[0,1],target_names=CLASSES,output_dict=True,zero_division=0)
    return {"accuracy":float(accuracy_score(yt,yp)),"macro_f1":float(f1_score(yt,yp,average="macro",zero_division=0)),
    "confusion_matrix":confusion_matrix(yt,yp,labels=[0,1]).tolist(),"classification_report":rep}

def run(exp,cfg,key,args):
    rep=REPS[key]; print("\n"+"="*70+f"\n{exp} | {key}\n"+"="*70)
    tr,ntr=dataset(args.metadata,"train",rep,args.batch_size,True,cfg["aug"],cfg["balance"])
    va,nva=dataset(args.metadata,"validation",rep,args.batch_size,False)
    model=build(key); model.compile(optimizer=keras.optimizers.Adam(args.learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    cw=weights(args.metadata) if cfg["cw"] else None
    d=args.output_dir/exp/key; d.mkdir(parents=True,exist_ok=True)
    cb=[keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=.001,patience=args.patience,restore_best_weights=True,verbose=1),
        keras.callbacks.ModelCheckpoint(d/"best.keras",monitor="val_loss",save_best_only=True,verbose=1),keras.callbacks.TerminateOnNaN()]
    t=time.time(); hist=model.fit(tr,validation_data=va,epochs=args.epochs,class_weight=cw,callbacks=cb,verbose=1); secs=time.time()-t
    model.save(d/"final.keras")
    te,nte=dataset(args.metadata,"test",rep,args.batch_size,False); ev=evaluate(model,te)
    hd={k:[float(v) for v in vs] for k,vs in hist.history.items()}; be=int(np.argmin(hd["val_loss"]))+1
    out={"experiment":exp,"model":key,"representation":rep,"config":cfg,"class_weights":cw,"train_records_presented":ntr,
         "validation_records":nva,"test_records":nte,"epochs_run":len(hd["val_loss"]),"best_epoch":be,
         "best_val_loss":hd["val_loss"][be-1],"training_seconds":secs,"test":ev,"history":hd}
    (d/"results.json").write_text(json.dumps(out,indent=2)+"\n")
    c=ev["classification_report"]["COVID-19"]
    print(f"TEST acc={ev['accuracy']:.4f} macroF1={ev['macro_f1']:.4f} COVID recall={c['recall']:.4f} COVID F1={c['f1-score']:.4f}")
    keras.backend.clear_session(); return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--metadata",type=Path,default=Path("data/coswara/coswara_heavy_split.csv"))
    ap.add_argument("--output-dir",type=Path,default=Path("data/coswara/imbalance_runs")); ap.add_argument("--epochs",type=int,default=100)
    ap.add_argument("--batch-size",type=int,default=32); ap.add_argument("--learning-rate",type=float,default=.001); ap.add_argument("--patience",type=int,default=10)
    ap.add_argument("--models",nargs="+",choices=list(REPS),default=list(REPS)); ap.add_argument("--experiments",nargs="+",choices=list(EXPS),default=list(EXPS))
    a=ap.parse_args(); results={}
    print("Validation/test remain original: no augmentation, no balancing.")
    for e in a.experiments:
        results[e]={}
        for m in a.models:
            try:results[e][m]=run(e,EXPS[e],m,a)
            except Exception as ex:
                print(f"FAILED {e}/{m}: {type(ex).__name__}: {ex}"); results[e][m]={"error":str(ex)}; keras.backend.clear_session()
    a.output_dir.mkdir(parents=True,exist_ok=True); sp=a.output_dir/"imbalance_experiments_summary.json"; sp.write_text(json.dumps(results,indent=2)+"\n")
    print(f"\nSummary: {sp}")
    for e,ms in results.items():
        print("\n"+e)
        for m,r in ms.items():
            if "test" in r:
                c=r["test"]["classification_report"]["COVID-19"]
                print(f"{m:8} acc={r['test']['accuracy']:.4f} macroF1={r['test']['macro_f1']:.4f} covidRecall={c['recall']:.4f} covidF1={c['f1-score']:.4f}")
            else:print(f"{m:8} FAILED")
if __name__=="__main__":main()
