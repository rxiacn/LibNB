/********************************************************************
* Naive Bayes Classifier V1.16
* Implemented by Rui Xia (rxia.cn@gmail.com)
* Last updated on 2012-01-09
*********************************************************************/

#include <cstdlib>
#include <iostream>
#include <string.h>
#include "NB.h"

using namespace std;


void print_help() {
    cout << "\nOpenPR-NB learning module\n\n"
        << "usage: nb_learn [options] training_file model_file\n\n"
        << "options: -h        -> help\n"
        << "         -e [0,1]  -> 0: multi-variate Bernoulli event model\n"
        << "                   -> 1: multinomial event model (default)\n"
        << "         -s [0]    -> Laplace smoothing (default)\n"
        << endl;
}

void read_parameters(int argc, char *argv[], char *training_file, char *model_file, 
                        int *event_model, int *smooth_tech) {
    // set default options
    *event_model = 1;
    *smooth_tech = 0;
    int i;
    for (i = 1; (i<argc) && (argv[i])[0]=='-'; i++) {
        switch ((argv[i])[1]) {
            case 'h':
                print_help();
                exit(0);
            case 'e':
                *event_model = atoi(argv[++i]);
                break;
            case 's':
                *smooth_tech = atoi(argv[++i]);
                break;
            default:
                cout << "Unrecognized option: " << argv[i] << "!" << endl;
                print_help();
                exit(0);
        }
    }
    
    if ((i+1)>=argc) {
        cout << "Not enough parameters!" << endl;
        print_help();
        exit(0);
    }
    strcpy (training_file, argv[i]);
    strcpy (model_file, argv[i+1]);
}

int nb_learn(int argc, char *argv[])
{
    char training_file[200];
    char model_file[200];
    int event_model;
    int smooth_tech;
    read_parameters(argc, argv, training_file, model_file, &event_model, &smooth_tech);
    NB nb;
    nb.load_training_file(training_file);
    nb.learn(event_model);
    nb.save_model(model_file);
    return 1;
}

int main(int argc, char *argv[])
{
    return nb_learn(argc, argv);
}
