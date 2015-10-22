/********************************************************************
* Naive Bayes Classifier V1.16
* Implemented by Rui Xia (rxia.cn@gmail.com)
* Last updated on 2012-01-09
*********************************************************************/

#include "NB.h"

NB::NB()
{
}

NB::~NB()
{
}

void NB::save_model(string model_file)
{ 
    cout << "Saving model..." << endl;
    ofstream fout(model_file.c_str());
    for (int j = 0; j < class_set_size; j++) {
        fout << class_prb[j] << " ";
    }
    fout << endl;
    for (int k = 0; k < feat_set_size; k++) {
        for (int j = 0; j < class_set_size; j++) {
            fout << feat_class_prb[k][j] << " ";
        }
        fout << endl;
    }
    fout.close();
}

void NB::load_model(string model_file)
{
    cout << "Loading model..." << endl;
    class_prb.clear();
    feat_class_prb.clear();
    ifstream fin(model_file.c_str());
    if(!fin) {
        cerr << "Error opening file: " << model_file << endl;
    }
    string line_str;
    // load class_prb
    getline(fin, line_str);
    vector<string> frist_line_vec = string_split(line_str, " ");
    for (vector<string>::iterator it = frist_line_vec.begin(); it != frist_line_vec.end(); it++) {
        float prb = (float)atof(it->c_str());
        class_prb.push_back(prb);
        
    }
    // load feat_class_prb
    while (getline(fin, line_str)) {
        vector<float> prb_vec;
        vector<string> line_vec = string_split(line_str, " ");
        for (vector<string>::iterator it = line_vec.begin(); it != line_vec.end(); it++) {
            float prb = (float)atof(it->c_str());
            prb_vec.push_back(prb);
        }
        feat_class_prb.push_back(prb_vec);
    }
    fin.close();
    feat_set_size = (int)feat_class_prb.size();
    class_set_size = (int)feat_class_prb[0].size();
}

void NB::read_samp_file(string samp_file, vector<sparse_feat> &samp_feat_vec, vector<int> &samp_class_vec) {
    ifstream fin(samp_file.c_str());
    if(!fin) {
        cerr << "Error opening file: " << samp_file << endl;
        exit(0);
    }
    string line_str;
    while (getline(fin, line_str)) {
        size_t class_pos = line_str.find_first_of("\t");
        int class_id = atoi(line_str.substr(0, class_pos).c_str());
		samp_class_vec.push_back(class_id);
        string terms_str = line_str.substr(class_pos+1);
        sparse_feat samp_feat;
		if (terms_str == "") { // in case of empty feature vector
			samp_feat.id_vec.push_back(1);
			samp_feat.value_vec.push_back(0);		
		}
		else {
			vector<string> fv_vec = string_split(terms_str, " ");
			for (vector<string>::iterator it = fv_vec.begin(); it != fv_vec.end(); it++) {
				size_t feat_pos = it->find_first_of(":");
				int feat_id = atoi(it->substr(0, feat_pos).c_str());
				int feat_value = (int)atof(it->substr(feat_pos+1).c_str());
				if (feat_value != 0) {
					samp_feat.id_vec.push_back(feat_id);
					samp_feat.value_vec.push_back(feat_value);              
				}
			}
		}
        samp_feat_vec.push_back(samp_feat);

    }
    fin.close();
}

void NB::load_training_file(string training_file)
{
    cout << "Loading training data..." << endl;
    read_samp_file(training_file, samp_feat_vec, samp_class_vec);
    feat_set_size = 0;
    class_set_size = 0;
    for (size_t i = 0; i < samp_class_vec.size(); i++) {
        if (samp_class_vec[i] > class_set_size) {
            class_set_size = samp_class_vec[i];
        }
        if (samp_feat_vec[i].id_vec.back() > feat_set_size) {
            feat_set_size = samp_feat_vec[i].id_vec.back();
        }   
    }
}

void NB::load_training_data(vector<sparse_feat> &feat_vec, vector<int> &class_vec)
{
	samp_feat_vec = feat_vec;
	samp_class_vec = class_vec;
    feat_set_size = (int)feat_class_prb.size();
    class_set_size = (int)feat_class_prb[0].size();
}

void NB::calc_class_prb()
{
    vector<int> class_freq(class_set_size, 0);
    for (vector<int>::iterator it_i = samp_class_vec.begin(); it_i != samp_class_vec.end(); it_i++) {
        int samp_class = *it_i;
        class_freq[samp_class-1]++;
    }
    for (int j = 0; j < class_set_size; j++) {
        class_prb.push_back((float)class_freq[j]/samp_class_vec.size());
    }
}

void NB::calc_feat_class_prb_multinomial()
{
    // count feat_class_tf
    vector< vector<int> > feat_class_tf;
    for (int k = 0; k < feat_set_size; k++) {
        vector<int> tf_vec(class_set_size, 0);
        feat_class_tf.push_back(tf_vec);
    }
    for (size_t i = 0; i < samp_feat_vec.size(); i++) {
        sparse_feat samp_feat = samp_feat_vec[i];
        int samp_class = samp_class_vec[i];
        for (size_t k = 0; k < samp_feat.id_vec.size(); k++) {
            int feat_id = samp_feat.id_vec[k];
            int feat_value = samp_feat.value_vec[k];
            feat_class_tf[feat_id-1][samp_class-1] += feat_value;
        }
    }
    // sum up class_tf
    vector<int> class_tf(class_set_size, 0);
    for (int j = 0; j < class_set_size; j++) {
        for (int k = 0; k < feat_set_size; k++) {
            class_tf[j] += feat_class_tf[k][j];
        }
    }
    // calculate feat_class_prb
    for (int k = 0; k < feat_set_size; k++) {
        vector<float> prb_vec;
        for (int j = 0; j < class_set_size; j++) {
            // freq to prb with Laplace smoothing
            prb_vec.push_back((float)(1 + feat_class_tf[k][j])/(feat_set_size + class_tf[j]));
        }
        feat_class_prb.push_back(prb_vec);
    }
}

void NB::calc_feat_class_prb_bernoulli()
{
    // count feat_class_df
    vector< vector<int> > feat_class_df;
    for (int k = 0; k < feat_set_size; k++) {
        vector<int> df_vec(class_set_size, 0);
        feat_class_df.push_back(df_vec);
    }
    for (size_t i = 0; i < samp_feat_vec.size(); i++) {
        sparse_feat samp_feat = samp_feat_vec[i];
        int samp_class = samp_class_vec[i];
        for (size_t k = 0; k < samp_feat.id_vec.size(); k++) {
            int feat_id = samp_feat.id_vec[k];
            int feat_value = samp_feat.value_vec[k];
            feat_class_df[feat_id-1][samp_class-1] += 1;
        }
    }
    // calculate class_df
    vector<int> class_df(class_set_size, 0);
    for (vector<int>::iterator it_i = samp_class_vec.begin(); it_i != samp_class_vec.end(); it_i++) {
        int samp_class = *it_i;
        class_df[samp_class-1]++;
    }
    // calculate feat_class_prb
    for (int k = 0; k < feat_set_size; k++) {
        vector<float> prb_vec;
        for (int j = 0; j < class_set_size; j++) {
            // freq to prb with Laplace smoothing
            prb_vec.push_back((float)(1 + feat_class_df[k][j])/(2 + class_df[j]));
        }
        feat_class_prb.push_back(prb_vec);
    }
}

void NB::learn(int event_model)
{
    cout << "Learning..." << endl;
    calc_class_prb();
    if (event_model == 0) {
        calc_feat_class_prb_bernoulli();
    }
    else {
        calc_feat_class_prb_multinomial();
    }
    samp_feat_vec.clear();
    samp_class_vec.clear();
}

vector<float> NB::predict_logp_bernoulli(sparse_feat &samp_feat)
{
    vector<int> feat_vec_out;
    int i = 0, k = 0;
    while (k < feat_set_size) {
        if ((k+1) != samp_feat.id_vec[i]) {
            feat_vec_out.push_back(k+1);
        }
        else if (i < (int)samp_feat.id_vec.size()-1) {
            i++;
        }
        k++;
    }
    vector<float> logp(class_set_size, 0);
    for (int j = 0; j < class_set_size; j++) {
        double logp_samp_given_class = 0.0;
        for (size_t k1 = 0; k1 < samp_feat.id_vec.size(); k1++) {
            int feat_id = samp_feat.id_vec[k1];
			if (feat_id > feat_set_size) // in case test samp have new feat id, 2011-11-18
				continue;
			logp_samp_given_class += log(feat_class_prb[feat_id-1][j]); 
        }
        for (size_t k0 = 0; k0 < feat_vec_out.size(); k0++) {
            int feat_id = feat_vec_out[k0];
			if (feat_id > feat_set_size) // in case test samp have new feat id, 2011-11-18
				continue;
			logp_samp_given_class += log(1-feat_class_prb[feat_id-1][j]);
        }
        double logp_samp_and_class = logp_samp_given_class + log(class_prb[j]);
        logp[j] = (float)logp_samp_and_class;
    }
    return logp;
}

vector<float> NB::predict_logp_multinomial(sparse_feat &samp_feat)
{
    vector<float> logp(class_set_size, 0);
    for (int j = 0; j < class_set_size; j++) {
        float logp_samp_given_class = 0.0;
        for (size_t k = 0; k < samp_feat.id_vec.size(); k++) {
            int feat_id = samp_feat.id_vec[k];
			if (feat_id > feat_set_size) // in case test samp have new feat id, 2011-11-18
				continue;
            int feat_value = samp_feat.value_vec[k];
            logp_samp_given_class += (log(feat_class_prb[feat_id-1][j]) * feat_value);
        }
        float logp_samp_and_class = logp_samp_given_class + log(class_prb[j]);
        logp[j] = logp_samp_and_class;
    }
    return logp;
}

vector<float> NB::score_to_prb(vector<float> &score)
{   
    vector<float> prb(class_set_size, 0);
    for (int i = 0; i < class_set_size; i++) {
        float delta_prb_sum = 0.0;
        for (int j = 0; j < class_set_size; j++) {
            delta_prb_sum += exp(score[j] - score[i]);
        }
        prb[i] = 1/delta_prb_sum;
    }
    return prb;
}

int NB::score_to_class(vector<float> &score)
{
    int pred_class = 0; 
    float max_score = score[0];
    for (int j = 1; j < class_set_size; j++) {
        if (score[j] > max_score) {
            max_score = score[j];
            pred_class = j;
        }
    }
    return ++pred_class;
}

float NB::classify_test_file(string test_file, string output_file, int event_model, int output_format)
{
    cout << "Classifying test file..." << endl;
    vector<sparse_feat> test_feat_vec;
    vector<int> test_class_vec;
    vector<int> pred_class_vec;
    read_samp_file(test_file, test_feat_vec, test_class_vec);
    ofstream fout(output_file.c_str());
    for (size_t i = 0; i < test_class_vec.size(); i++) {
        int samp_class = test_class_vec[i];
        sparse_feat samp_feat = test_feat_vec[i];
        vector<float> pred_score;
        if (event_model == 0) {
            pred_score = predict_logp_bernoulli(samp_feat);
        }
        else {
            pred_score = predict_logp_multinomial(samp_feat);
        }       
        int pred_class = score_to_class(pred_score);
        pred_class_vec.push_back(pred_class);
        fout << pred_class << "\t";
        if (output_format == 1) {
            for (int j = 0; j < class_set_size; j++) {
                fout << pred_score[j] << ' '; 
            }       
        }
        else if (output_format == 2) {
            vector<float> pred_prb = score_to_prb(pred_score);
            for (int j = 0; j < class_set_size; j++) {
                fout << pred_prb[j] << ' '; 
            }
        }
        fout << endl;       
    }
    fout.close();
    float acc = calc_acc(test_class_vec, pred_class_vec);
    return acc; 
}

float NB::classify_test_data(vector<sparse_feat> &test_feat_vec, vector<int> &test_class_vec, vector<int> &pred_class_vec, vector< vector<float> > pred_prb_vec, int event_model)
{
    for (size_t i = 0; i < test_class_vec.size(); i++) {
        int samp_class = test_class_vec[i];
        sparse_feat samp_feat = test_feat_vec[i];
        vector<float> pred_score;
        if (event_model == 0) {
            pred_score = predict_logp_bernoulli(samp_feat);
        }
        else {
            pred_score = predict_logp_multinomial(samp_feat);
        }
        int pred_class = score_to_class(pred_score);
		vector<float> pred_prb = score_to_prb(pred_score);
		pred_class_vec.push_back(pred_class);
		pred_prb_vec.push_back(pred_prb);
    }
    float acc = calc_acc(test_class_vec, pred_class_vec);
    return acc; 
}


float NB::calc_acc(vector<int> &test_class_vec, vector<int> &pred_class_vec)
{
    size_t len = test_class_vec.size();
    if (len != pred_class_vec.size()) {
        cerr << "Error: two vectors should have the same lenght." << endl;
        exit(0);
    }
    int err_num = 0;
    for (size_t id = 0; id != len; id++) {
        if (test_class_vec[id] != pred_class_vec[id]) {
            err_num++;
        }
    }
    return 1 - ((float)err_num) / len;
}

vector<string> NB::string_split(string terms_str, string spliting_tag)
{
    vector<string> feat_vec;
    size_t term_beg_pos = 0;
    size_t term_end_pos = 0;
    while ((term_end_pos = terms_str.find_first_of(spliting_tag, term_beg_pos)) != string::npos) {
        if (term_end_pos > term_beg_pos) {
            string term_str = terms_str.substr(term_beg_pos, term_end_pos - term_beg_pos);
            feat_vec.push_back(term_str);
        }
        term_beg_pos = term_end_pos + 1;
    }
    if (term_beg_pos < terms_str.size()) {
        string end_str = terms_str.substr(term_beg_pos);
        feat_vec.push_back(end_str);
    }
    return feat_vec;
}
