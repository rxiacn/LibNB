/********************************************************************
* Naive Bayes Classifier V1.15
* Implemented by Rui Xia (rxia.cn@gmail.com)
* Last updated on 2011-11-18
*********************************************************************/

#include <cstdlib>
#include <iostream>
#include <string.h>
#include "NB.h"

using namespace std;


void print_help() {
    cout << "\nOpenPR-NB classification module\n\n"
        << "usage: nb_classify [options] test_file model_file output_file\n\n"
        << "options: -h        -> help\n"
        << "         -e [0,1]  -> 0: multi-variate Bernoulli event model\n"
        << "                   -> 1: multinomial event model (default)\n"       
        << "         -f [0..2] -> 0: only output class label (default)\n"
        << "                   -> 1: output class label with log-likelihood\n"
        << "                   -> 2: output class label with probability\n"
        << endl;
}

void read_parameters(int argc, char *argv[], char *test_file, char *model_file, 
                        char *output_file, int *event_model, int *output_format) {
    // set default options
    *output_format = 0;
    *event_model = 1;
    int i;
    for (i = 1; (i<argc) && (argv[i])[0]=='-'; i++) {
        switch ((argv[i])[1]) {
            case 'h':
                print_help();
                exit(0);
            case 'e':
                *event_model = atoi(argv[++i]);
                break;
            case 'f':
                *output_format = atoi(argv[++i]);
                break;
            default:
                cout << "Unrecognized option: " << argv[i] << "!" << endl;
                print_help();
                exit(0);
        }
    }
    
    if ((i+2)>=argc) {
        cout << "Not enough parameters!" << endl;
        print_help();
        exit(0);
    }
    strcpy(test_file, argv[i]);
    strcpy(model_file, argv[i+1]);
    strcpy(output_file, argv[i+2]);
}

int main(int argc, char *argv[])
{
    char test_file[200];
    char model_file[200];
    char output_file[200];
    int event_model;
    int output_format;
    read_parameters(argc, argv, test_file, model_file, output_file, &event_model, &output_format);
    NB nb;
    nb.load_model(model_file);
    float acc= nb.classify_test_file(test_file, output_file, event_model, output_format);
    cout << "Accuracy: " << acc << endl;
    return 1;
}
