package cz.zcu.kiv.nlp.ir.trec;

import cz.zcu.kiv.nlp.ir.trec.data.*;
import cz.zcu.kiv.nlp.ir.trec.preprocessing.*;

import java.lang.reflect.Array;
import java.util.*;

/**
 * @author tigi
 */

public class Index implements Indexer, Searcher {

    private boolean toLowercase;
    private boolean removeAccentsBeforeStemming;
    private boolean removeAccentsAfterStemming;

    private Stemmer stemmer;
    private Tokenizer tokenizer;
    private Set<String> stopwords;

    public Map<String, Map<String, Posting>> invertedIndex = new HashMap<String, Map<String, Posting>>();

    Map<String, Document> documents = new HashMap<String, Document>();

    Index() {
        this(new AdvancedTokenizer(), new CzechStemmerAgressive(), false, true, true);
    }

    Index(Tokenizer tokenizer, Stemmer stemmer, boolean removeAccentsBeforeStemming, boolean removeAccentsAfterStemming, boolean toLowercase) {
        this.tokenizer = tokenizer;
        this.stemmer = stemmer;

        this.removeAccentsAfterStemming = removeAccentsAfterStemming;
        this.removeAccentsBeforeStemming = removeAccentsBeforeStemming;
        this.toLowercase = toLowercase;

        stopwords = StopwordsReader.Read();
    }


//region index
    public void index(List<Document> documents) {
        for (Document document : documents) {
            processDocument(document);
            this.documents.put(document.getId(), document);
        }
        computeTfIdf();
    }

    public void index(Document document){
        processDocument(document);
        this.documents.put(document.getId(), document);

        computeTfIdf();
    }

    public void update(Document document){
        delete(document);
        index(document);

        computeTfIdf();
    }

    public void delete(Document document){
        for(Map<String, Posting> res : invertedIndex.values()){
            res.remove(document.getId());
        }
        this.documents.remove(document);

        computeTfIdf();
    }


    private void processDocument(Document document) {
        List<String> tokens = new ArrayList<String>();

        tokens.addAll(processString(document.getTitle()));
        tokens.addAll(processString(document.getText()));
        tokens.addAll(processString(document.getDate().toString()));

        document.setSize(tokens.size());

        for (String token : tokens) {
            if(invertedIndex.containsKey(token)){
                if(invertedIndex.get(token).containsKey(document.getId())){
                    invertedIndex.get(token).get(document.getId()).increaseFrequency();
                }
                else{
                    invertedIndex.get(token).put(document.getId(), new Posting(document.getId()));
                }
            }
            else{
                invertedIndex.put(token, new HashMap<String, Posting>());
                invertedIndex.get(token).put(document.getId(), new Posting(document.getId()));
            }
        }
    }

    private List<String> processString(String text) {
        List<String> tokens = new ArrayList<String>();

        if (toLowercase) {
            text = text.toLowerCase();
        }
        if (removeAccentsBeforeStemming) {
            text = AdvancedTokenizer.removeAccents(text);
        }
        for (String token : tokenizer.tokenize(text)) {

            token = stemmer.stem(token);

            if (removeAccentsAfterStemming) {
                token = AdvancedTokenizer.removeAccents(token);
            }
            if (stopwords.contains(token)) {
                continue;
            }
            tokens.add(token);
        }
        return tokens;
    }

    private void computeTfIdf(){
        for(String term : invertedIndex.keySet()){
            double idf = Math.log10((double)documents.size()/invertedIndex.get(term).size());

            for(Posting res : invertedIndex.get(term).values()){
                double tf = (double)res.getFrequency() / documents.get(res.getDocumentID()).getSize();

                res.setTfIdf(tf*idf);
                //res.setTfIdf((1+Math.log10(tf))*idf);
            }
        }
    }

//endregion

//region search
    public List<Result> search(String query_in) {
        List<Result> result = new ArrayList<Result>();

        List<String> query = processString(query_in);
        //<operand, terms>
        Map<String, List<String>> logic = getLogic(query);
        query.removeAll(logic.keySet());    //remove AND/OR/NOT

        double[] tfidf = computeTfIdfQuery(query);
        double wq = normalizeVector(tfidf);

        //<docID, score>
        Map<String, Double> wd = getNormalizedWeights(query, logic);

        for(String docId : wd.keySet()){
            double score = wq*wd.get(docId);
            result.add(new ResultImpl(docId, score));
        }

        Collections.sort(result, (r1,r2)->((Double)r2.getScore()).compareTo(r1.getScore())  );

        return result;
    }

    private Map<String, List<String>> getLogic(List<String> query){
        Map<String, List<String>> logic = new HashMap<String, List<String>>();
        logic.put("AND", new ArrayList<String>());
        logic.put("OR", new ArrayList<String>());
        logic.put("NOT", new ArrayList<String>());

        for (int i = 1; i < query.size() -1; i++){
            if(query.get(i).equals("and")){
                logic.get("AND").add(query.get(i-1));
                logic.get("AND").add(query.get(i+1));
            }
            if(query.get(i).equals("or")){
                logic.get("OR").add(query.get(i-1));
                logic.get("OR").add(query.get(i+1));
            }
            if(query.get(i).equals("not")){
                logic.get("NOT").add(query.get(i+1));
            }
        }
        return logic;
    }

    private double[] computeTfIdfQuery(List<String> query){
        double[] tfidf = new double[query.size()];

        for(int i = 0; i < query.size(); i++){
            //for(String term : query){
            if(!invertedIndex.containsKey(query.get(i))) continue;

            double idf = Math.log10((double)documents.size()/invertedIndex.get(query.get(i)).size());
            double tf = 1.0 / query.size();

            tfidf[i] = tf*idf;
        }
        return tfidf;
    }

    private Map<String, Double> getNormalizedWeights(List<String> query, Map<String, List<String>> logic) {
        //get weight vector
        //<docID, vector>
        Map<String, double[]> weights = new HashMap<String, double[]>();

        for (int i = 0; i < query.size(); i++) {
            if (!invertedIndex.containsKey(query.get(i))) continue;

            for (Posting post : invertedIndex.get(query.get(i)).values()) {
                if (!weights.containsKey(post.getDocumentID())) {
                    weights.put(post.getDocumentID(), new double[query.size()]);
                }
                weights.get(post.getDocumentID())[i] = post.getTfIdf();
            }
        }

        //apply logic
        List<String> removeList = new ArrayList<String>();

        for (String docId : weights.keySet()) {
            double[] vector = weights.get(docId);

            List<String> exp = logic.get("AND");
            for (int i = 0; i < exp.size(); i += 2) {
                int pos1 = query.indexOf(exp.get(i));
                int pos2 = query.indexOf(exp.get(i+1));
                if (vector[pos1] == 0 || vector[pos2] == 0){
                    //weights.remove(docId);
                    removeList.add(docId);
                    continue;
                }
            }
            exp = logic.get("OR");
            for (int i = 0; i < exp.size(); i += 2) {
                int pos1 = query.indexOf(exp.get(i));
                int pos2 = query.indexOf(exp.get(i+1));
                if (vector[pos1] == 0 && vector[pos2] == 0){
                    //weights.remove(docId);
                    removeList.add(docId);
                    continue;
                }
            }
            exp = logic.get("NOT");
            for (int i = 0; i < exp.size(); i++) {
                int pos = query.indexOf(exp.get(i));
                if (vector[pos] > 0){
                    //weights.remove(docId);
                    removeList.add(docId);
                    continue;
                }
            }
        }
        //remove only if something left
        if(removeList.size() < weights.size()){
            //weights.remove(removeList);
            for(String docId : removeList){
                weights.remove(docId);
            }
        }

        //normalize vectors
        //<docId, score>
        Map<String, Double> wd = new HashMap<String, Double>();

        for(String docId : weights.keySet()){
            double w = normalizeVector(weights.get(docId));
            wd.put(docId, w);
        }

        return wd;
    }

    private double normalizeVector(double[] vector) {
        double sum = 0.0;
        for (double x : vector) {
            sum += x * x;
        }
        return Math.sqrt(sum);
    }

    //endregion
}
