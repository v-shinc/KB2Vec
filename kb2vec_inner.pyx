#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8


import cython
import numpy as np
cimport numpy as np

from libc.math cimport exp
from libc.string cimport memset



cdef extern from "voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)

cdef extern from "cblas.h":
    enum CBLAS_ORDER:
        CblasRowMajor=101
        CblasColMajor=102

from scipy.linalg.blas import fblas


ctypedef np.float32_t REAL_t
DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6
DEF MAX_SIZE = 200
cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0
cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE


def train_triple_sg(model,triple,alpha,_work,_detR):
    cdef int hs = model.hs
    cdef int negative = model.negative

    cdef REAL_t *syn1
    cdef int codelens
    cdef np.uint32_t *point
    cdef np.uint8_t *codes

    cdef np.uint32_t *table
    cdef unsigned long long table_len
    cdef unsigned long long next_random

    cdef REAL_t _alpha = alpha
    cdef int size = model.layer1_size
    e1,r,e2 = triple[0],triple[1],triple[2]

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *R = <REAL_t *>(np.PyArray_DATA(model.rel_mat))

    cdef np.uint32_t e1_index = e1.index
    cdef np.uint32_t r_index = r.index
    cdef np.uint32_t e2_index = e2.index


    cdef REAL_t *work = <REAL_t*>(np.PyArray_DATA(_work))
    cdef REAL_t *detR = <REAL_t*>(np.PyArray_DATA(_detR))

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))
        codelens = <int> len(e2.code)
        point = <np.uint32_t *>np.PyArray_DATA(e2.point)
        codes = <np.uint8_t *>np.PyArray_DATA(e2.code)


    # For negative sampling
    if negative:
        table = <np.uint32_t*>(np.PyArray_DATA(model.table))
        table_len = len(model.table)
        next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)


    with nogil:
        if hs:
            fast_triple_sg_hs(point,codes,codelens, syn0, syn1, size,e1_index, R, r_index, _alpha, work,detR)
        if negative:
            fast_triple_sg_neg(negative, table, table_len,syn0, size, e1_index, R, r_index, e2_index, _alpha, work, detR, next_random)

    return 1


cdef void fast_triple_sg_hs(
    const np.uint32_t *e2_point, const np.uint8_t *e2_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1, const int size,const np.uint32_t e1_index,
    REAL_t *R, const np.uint32_t r_index,
    const REAL_t alpha, REAL_t *work, REAL_t *detR) nogil:

    cdef long long i,j,k
    cdef long long row1 = e1_index * size, row2 = r_index * size * size, row3
    cdef REAL_t f,g

    memset(detR,0,size*size*cython.sizeof(REAL_t))
    #memset(work,0,size*np.cython(REAL_t))
    for a in range(size):
        work[a] = <REAL_t>0.0

    for j in range(codelen):
        row3 = e2_point[j] * size
        for k in range(size):
            for i in range(size):
                f += syn0[row1 + i] * R[row2 + i*size + k] * syn1[row3 + k]

        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

        g = (1 - e2_code[j] - f) * alpha

        for i in range(size):
            for k in range(size):
                detR[i*size + k] += g * syn0[row1 + i] * syn1[row3 + k]
        for i in range(size):
            for k in range(size):
                work[i] += g * R[row2 + i *size + k] * syn1[row3 + k]
        for k in range(size):
            for i in range(size):
                syn1[row3 + k] += g * syn0[row1 + i] * R[row2 + i * size + k]
    for i in range(size):
        syn0[row1 + i] += work[i]
    for i in range(size):
        for k in range(size):
            R[row2 + i * size + k] = detR[i * size + k]


cdef unsigned long long fast_triple_sg_neg(
    const int negative, np.uint32_t *table, unsigned long long table_len,
    REAL_t *syn0, const int size, const np.uint32_t e1_index,
    REAL_t * R, const np.uint32_t r_index,
    const np.uint32_t e2_index, const REAL_t alpha, REAL_t *work, REAL_t *detR,
    unsigned long long next_random) nogil:

    cdef long long i,j,k
    cdef long long row1 = e1_index * size, row2 = r_index * size, row3
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label
    cdef np.uint32_t target_index


    for i in range(size):
        work[i] = <REAL_t> 0.0

    for i in range(size):
        for k in range(size):
            detR[i * size + k] = <REAL_t> 0.0

    for j in range(negative + 1):
        if j == 0:
            target_index = e2_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == e2_index:
                continue
            label = <REAL_t>0.0
        row3 = target_index * size
        f = <REAL_t> 0.0
        for k in range(size):
            for i in range(size):
                f += syn0[row1 + i] * R[row2 + i * size + k] * syn0[row3 + k]
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        for i in range(size):
            for k in range(size):
                detR[i * size + k] += g * syn0[row1 + i] * syn0[row3 + k]

        for i in range(size):
            for k in range(size):
                work[i] += g * R[row2 + i * size +k] * syn0[row3 + k]
        for k in range(size):
            for i in range(size):
                syn0[row3 + k] += g * syn0[row1 + i] * R[row2 + i * size + k]

    for i in range(size):
        syn0[row1 + i] += work[i]

    for i in range(size):
        for k in range(size):
            R[row2 + i * size + k] = detR[i * size + k]

def init():
    cdef int i
    #build the sigmoid table
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = <REAL_t>exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))
    return 1
FAST_VERSION = init()