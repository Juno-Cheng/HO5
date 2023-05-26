#!/usr/bin/env python3

"""
Do a local practice grading.
The score you recieve here is not an actual score,
but gives you an idea on how prepared you are to submit to the autograder.
"""

import os
import sys

import pandas
import numpy
import sklearn.dummy

import cse40.question
import cse40.assignment
import cse40.style
import cse40.utils

THIS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(THIS_DIR, 'data.txt')

class T1A(cse40.question.Question):
    def score_question(self, submission, data):
        result = submission.clean_data(data)
        if (self.check_not_implemented(result)):
            return

        if (not isinstance(result, pandas.DataFrame)):
            self.fail("Answer must be a DataFrame.")
            return

        self.full_credit()

class T3A(cse40.question.Question):
    def score_question(self, submission, data):
        result = submission.create_classifiers()
        if (self.check_not_implemented(result)):
            return

        if (not isinstance(result, list)):
            self.fail("Answer must be a list.")
            return

        if (len(result) != 3):
            self.fail("Answer must be a list with three elements.")
            return

        self.full_credit()

class T3B(cse40.question.Question):
    def score_question(self, submission, data):
        folds = 5
        result = submission.cross_fold_validation(sklearn.dummy.DummyClassifier(), data, folds)
        if (self.check_not_implemented(result)):
            return

        if (not isinstance(result, list)):
            self.fail("Answer must be a list.")
            return

        if (len(result) != folds):
            self.fail("Answer must be a list with the same number of elements as folds.")
            return

        self.full_credit()

class T3C(cse40.question.Question):
    def score_question(self, submission, data):
        result = submission.significance_test([1, 2, 3], [1, 2, 3], 0.1)
        if (self.check_not_implemented(result)):
            return

        if (not isinstance(result, (bool, numpy.bool_))):
            self.fail("Answer must be a boolean.")
            return

        self.full_credit()

def grade(path):
    submission = cse40.utils.prepare_submission(path)
    additional_data = {
        'data': pandas.read_csv(DATA_PATH, sep = "\t")
    }

    questions = [
        T1A("Task 1.A (clean_data)", 1),
        T3A("Task 3.A (create_classifiers)", 1),
        T3B("Task 3.B (cross_fold_validation)", 1),
        T3C("Task 3.C (significance_test)", 1),
        cse40.style.Style(path, max_points = 1),
    ]

    assignment = cse40.assignment.Assignment('Practice Grading for Hands-On 5', questions)
    assignment.grade(submission, additional_data = additional_data)

    return assignment

def main(path):
    assignment = grade(path)
    print(assignment.report())

def _load_args(args):
    exe = args.pop(0)
    if (len(args) != 1 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args})):
        print("USAGE: python3 %s <submission path (.py or .ipynb)>" % (exe), file = sys.stderr)
        sys.exit(1)

    path = os.path.abspath(args.pop(0))

    return path

if (__name__ == '__main__'):
    main(_load_args(list(sys.argv)))
