import {normalizeText, cosineScores, rankScores, cosineBreakdown} from './retrieval.js';
const status=document.getElementById('status'),output=document.getElementById('output'),button=document.getElementById('run');
const assert=(condition,message)=>{if(!condition)throw new Error(message);};
const log=message=>{output.textContent+=message+'\n';};
button.addEventListener('click',async()=>{
  button.disabled=true;output.textContent='';status.textContent='Menyiapkan pemeriksaan…';
  const worker=new Worker(new URL('./search-engine.js',import.meta.url),{type:'module'});
  function request(payload,expected){
    return new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>{worker.removeEventListener('message',listen);reject(new Error('Waktu pemeriksaan habis (180 detik).'));},180000);
      const listen=({data})=>{
        if(['loading','stage'].includes(data.type))status.textContent=data.message;
        if(data.type===expected||(data.type==='error'&&expected!=='error')){
          clearTimeout(timer);worker.removeEventListener('message',listen);
          if(data.type==='error')reject(new Error(data.message));else resolve(data);
        }
      };
      worker.addEventListener('message',listen);worker.postMessage(payload);
    });
  }
  worker.onerror=event=>{status.textContent='Runtime gagal: '+event.message;};
  try{
    assert(normalizeText('  science\n search  ')==='science search','Normalisasi gagal.');
    const scores=cosineScores([1,0],[0,1,1,0,1,0],2),ranked=rankScores(scores);
    assert(ranked[0].index===1&&ranked[1].index===2,'Ranking atau skor seri tidak stabil.');
    assert(cosineBreakdown([1,0],[1,0],0,2).score===1,'Penjumlahan cosine gagal.');
    log('PASS · Normalisasi, cosine, dan pengurutan skor seri.');
    const result=await request({type:'verify'},'verified');
    for(const item of result.results){
      assert(item.maximumError<1e-4,'Selisih embedding melebihi toleransi.');
      assert(item.top5Matches,'Top-5 berbeda dari acuan Python.');
      log(`PASS · ${item.model} · embedding error ${item.maximumError.toExponential(2)} · Top-5 cocok · ${item.query}`);
    }
    const compared=await request({type:'search',variant:'compare',query:'The PPR MDA5 has two N-terminal CARD domains.',topK:5},'comparison');
    assert(compared.pretrained.model==='pretrained'&&compared.finetuned.model==='finetuned','Identitas model tercampur.');
    assert(compared.finetuned.results[0].id==='31272411','Ranking final query MDA5 berbeda.');
    assert(compared.finetuned.queryMetrics['nDCG@10']===1,'Metrik query benchmark final tidak sesuai.');
    log('PASS · Mode perbandingan, pasangan encoder/corpus, dan metrik benchmark.');
    const detailed=await request({type:'detail',variant:'finetuned',index:compared.finetuned.results[0].index},'detail');
    assert(Math.abs(detailed.breakdown.score-compared.finetuned.results[0].score)<1e-8,'Rincian cosine tidak sama dengan ranking.');
    log('PASS · Total kontribusi dimensi sama dengan skor ranking.');
    const long=await request({type:'search',variant:'finetuned',query:'Scientific evidence '.repeat(200),topK:5},'result');
    assert(long.data.truncated&&long.data.usedTokens===256,'Truncation tidak bekerja.');
    assert(Math.abs(long.data.normAfter-1)<1e-5,'Embedding tidak ternormalisasi.');
    log('PASS · Input panjang dipotong ke 256 token; norma vektor 1.');
    const empty=await request({type:'search',variant:'finetuned',query:'  ',topK:5},'error');
    assert(empty.message.includes('Tulis'),'Query kosong tidak ditolak.');
    log('PASS · Query kosong ditolak.');
    status.textContent='SEMUA PEMERIKSAAN LULUS';
  }catch(error){log('FAIL · '+error.message);status.textContent='Ada pemeriksaan yang belum lulus.';}
  finally{worker.terminate();button.disabled=false;}
});
