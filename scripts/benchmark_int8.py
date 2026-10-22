import copy,json,time,warnings
from pathlib import Path
import numpy as np, torch
from timestral import TimestralForecaster
from infer import load_prices, ROOT
warnings.filterwarnings('ignore',category=DeprecationWarning)
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
m=json.loads((ROOT/'data/manifest.json').read_text())
f=TimestralForecaster.from_pretrained('mistralai/Timestral-209M',revision=m['model_revision'],local_files_only=True)
x=[]
for ticker in ['SPY','RELIANCE.NS','EURUSD=X']:
 r=np.diff(np.log(load_prices(ticker).to_numpy()))*100
 for end in [-1,-50,-150,-300]: x.append(r[len(r)+end-249:len(r)+end+1])
x=np.array(x,dtype='float32')
t=time.perf_counter(); ref=f.forecast(x,20,clip_nonnegative=False).quantiles.numpy(); print('fp32 12',time.perf_counter()-t,flush=True)
spec={name:torch.ao.quantization.default_dynamic_qconfig for name,module in f.model.named_modules() if isinstance(module,torch.nn.Linear) and not name.startswith('representation')}
f.model=torch.ao.quantization.quantize_dynamic(f.model,spec,dtype=torch.qint8,inplace=True)
for context,batch in [(250,12),(100,64),(250,64),(500,64)]:
 xx=x if batch==12 else np.random.default_rng(42).normal(0,1,(batch,context)).astype('float32')
 t=time.perf_counter(); out=f.forecast(xx,20,clip_nonnegative=False).quantiles.numpy(); print('int8',context,batch,time.perf_counter()-t,flush=True)
 if batch==12:
  width=ref[:,:,:,18]-ref[:,:,:,2]
  report={'max_quantile_difference_pct_points':float(np.max(abs(out-ref))), 'median_abs_quantile_difference_pct_points':float(np.median(abs(out-ref))), 'median_abs_difference_divided_80_width':float(np.median(abs(out-ref)/np.maximum(width[...,None],1e-8))), 'max_abs_difference_divided_80_width':float(np.max(abs(out-ref)/np.maximum(width[...,None],1e-8))), 'median_sign_agreement':float(np.mean(np.sign(out[:,:,:,10])==np.sign(ref[:,:,:,10])))}
  print(json.dumps(report),flush=True)
  (ROOT/'results/int8_validation.json').write_text(json.dumps(report,indent=2))
