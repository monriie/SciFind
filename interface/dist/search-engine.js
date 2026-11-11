import { pipeline, env } from './vendor/transformers.js';
import { normalizeText, cosineScores, rankScores, scoreHistogram, cosineBreakdown } from './retrieval.js';
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = new URL('./models/', import.meta.url).href;
env.backends.onnx.wasm.wasmPaths = new URL('./vendor/ort/', import.meta.url).href;
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = false;
let documents, manifest, examples, commonPromise;
const models={}, vectors={}, pending={}, lastVectors={};
const send=(type,payload={})=>postMessage({type,...payload});
async function get(path){
  const response=await fetch(new URL(path,import.meta.url));
  if(!response.ok)throw new Error(`File ${path} tidak tersedia (HTTP ${response.status}).`);
  return response;
}
async function initialize(variant){
  if(!['pretrained','finetuned'].includes(variant))throw new Error('Model tidak valid.');
  if(models[variant])return;
  if(!pending[variant])pending[variant]=(async()=>{
    send('loading',{message:'Membaca corpus dan embedding…',progress:4});
    if(!commonPromise)commonPromise=Promise.all([
      get('./data/manifest.json').then(r=>r.json()),get('./data/documents.json').then(r=>r.json()),
      get('./data/examples.json').then(r=>r.json())
    ]).then(values=>{[manifest,documents,examples]=values;});
    await commonPromise;
    const config=manifest.variants[variant];
    vectors[variant]=new Float32Array(await get('./data/'+config.embeddings).then(r=>r.arrayBuffer()));
    if(vectors[variant].length!==documents.length*384)throw new Error('Dimensi embedding tidak cocok.');
    send('loading',{message:`Memuat model ${config.label} (±91 MB)…`,progress:15});
    models[variant]=await pipeline('feature-extraction',config.model,{device:'wasm',dtype:'fp32',
      progress_callback:event=>{
        if(event.status==='progress'&&event.file?.endsWith('.onnx'))
          send('loading',{message:`Memuat model ${config.label}…`,progress:15+event.progress*.8});
      }});
    send('ready',{manifest,variant});
  })();
  try{await pending[variant];}catch(error){delete pending[variant];throw error;}
}
async function encode(variant,text){
  const features=models[variant].tokenizer(text,{truncation:true,max_length:256});
  const output=await models[variant].model(features),hidden=output.last_hidden_state;
  if(!hidden||hidden.dims[2]!==384)throw new Error('Output encoder tidak sesuai MiniLM.');
  const mask=Array.from(features.attention_mask.data,Number),active=mask.reduce((a,b)=>a+b,0);
  const vector=new Float32Array(384);
  for(let d=0;d<384;d++){
    let sum=0;for(let t=0;t<mask.length;t++)sum+=hidden.data[t*384+d]*mask[t];
    vector[d]=sum/active;
  }
  const normBefore=Math.sqrt(vector.reduce((sum,v)=>sum+v*v,0));
  for(let d=0;d<384;d++)vector[d]/=normBefore;
  return{vector,features,normBefore};
}
async function search(query,topK,variant){
  await initialize(variant);
  const started=performance.now(),normalized=normalizeText(query);
  if(!normalized)throw new Error('Tulis kalimat pencarian terlebih dahulu.');
  if(normalized.length>12000)throw new Error('Query maksimal 12.000 karakter.');
  if(![5,10,20].includes(topK))throw new Error('Jumlah hasil tidak valid.');
  send('stage',{message:`${manifest.variants[variant].label}: tokenisasi dan embedding query…`});
  const allTokens=models[variant].tokenizer(normalized,{truncation:false});
  const encodeStarted=performance.now();
  const {vector,features,normBefore}=await encode(variant,normalized);
  const encodingMs=performance.now()-encodeStarted;
  const ids=Array.from(features.input_ids.data,Number);
  const tokens=ids.map(id=>({id,text:models[variant].tokenizer.decode([id],{skip_special_tokens:false})}));
  send('stage',{message:`${manifest.variants[variant].label}: menghitung cosine seluruh corpus…`});
  const rankingStarted=performance.now();
  const scores=cosineScores(vector,vectors[variant]),ranking=rankScores(scores);
  const rankingMs=performance.now()-rankingStarted;
  const benchmark=examples.find(e=>normalizeText(e.text)===normalized);
  const results=ranking.slice(0,topK).map(({index,score},i)=>({...documents[index],index,score,rank:i+1,variant,
    relevant:benchmark?benchmark.relevant.includes(documents[index].id):null}));
  lastVectors[variant]=vector;
  let queryMetrics=null;
  if(benchmark){
    const hits=ranking.slice(0,100).map(r=>Number(benchmark.relevant.includes(documents[r.index].id)));
    const dcg=hits.slice(0,10).reduce((s,h,i)=>s+h/Math.log2(i+2),0);
    const ideal=Array.from({length:Math.min(10,benchmark.relevant.length)},(_,i)=>1/Math.log2(i+2)).reduce((a,b)=>a+b,0);
    let found=0,ap=0;hits.forEach((hit,i)=>{found+=hit;if(hit)ap+=found/(i+1);});
    queryMetrics={'nDCG@10':dcg/ideal,'AP@100':ap/benchmark.relevant.length,'Recall@100':found/benchmark.relevant.length};
  }
  return{original:query,normalized,model:variant,modelLabel:manifest.variants[variant].label,tokens,
    tokenCount:allTokens.input_ids.dims[1],usedTokens:ids.length,truncated:allTokens.input_ids.dims[1]>256,
    vector:Array.from(vector),normBefore,normAfter:Math.sqrt(vector.reduce((sum,v)=>sum+v*v,0)),
    encodingMs,rankingMs,totalMs:performance.now()-started,results,topK,documents:documents.length,queryMetrics,
    histogram:scoreHistogram(scores),
    benchmark:benchmark?{id:benchmark.id,relevantCount:benchmark.relevant.length,relevantFound:results.filter(r=>r.relevant).length}:null,
    allRanking:ranking.slice(0,100).map(({index,score},i)=>({rank:i+1,id:documents[index].id,title:documents[index].title,score})),
  };
}
let working=false;
self.onmessage=async({data})=>{
  if(working){send('error',{message:'Tunggu proses yang sedang berjalan selesai.'});return;}
  working=true;
  try{
    if(data.type==='search'){
      if(data.variant==='compare'){
        const pretrained=await search(data.query,data.topK,'pretrained');
        const finetuned=await search(data.query,data.topK,'finetuned');
        send('comparison',{pretrained,finetuned});
      }else send('result',{data:await search(data.query,data.topK,data.variant||'finetuned')});
    }else if(data.type==='detail'){
      const variant=data.variant||'finetuned';
      if(!lastVectors[variant]||!Number.isInteger(data.index)||!documents[data.index])throw new Error('Pilih hasil yang tersedia.');
      send('detail',{index:data.index,variant,breakdown:cosineBreakdown(lastVectors[variant],vectors[variant],data.index)});
    }else if(data.type==='verify'){
      const results=[];
      for(const variant of ['pretrained','finetuned']){
        await initialize(variant);
        const fixtures=await get(`./data/parity-fixtures-${variant}.json`).then(r=>r.json());
        for(const fixture of fixtures){
          const {vector}=await encode(variant,fixture.query);
          const maximumError=Math.max(...Array.from(vector,(v,i)=>Math.abs(v-fixture.embedding[i])));
          const ranked=rankScores(cosineScores(vector,vectors[variant]));
          results.push({model:manifest.variants[variant].label,query:fixture.query,maximumError,
            top5Matches:ranked.slice(0,5).every((row,i)=>documents[row.index].id===fixture.top5[i].id)});
        }
      }
      send('verified',{results});
    }
  }catch(error){send('error',{message:error.message||String(error)});}
  finally{working=false;}
};
