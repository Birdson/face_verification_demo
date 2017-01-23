/*
 * Copyright 2009-2016 Pegatron Corporation. All Rights Reserved.
 *
 * Pegatron Corporation. Confidential and Proprietary
 *
 * The following software source code ("Software") is strictly confidential and
 * is proprietary to Pegatron Corporation. ("PEGATRON").  It may only be read,
 * used, copied, adapted, modified or otherwise dealt with by you if you have
 * entered into a confidentiality agreement with PEGATRON and then subject to the
 * terms of that confidentiality agreement and any other applicable agreement
 * between you and PEGATRON.  If you are in any doubt as to whether you are
 * entitled to access, read, use, copy, adapt, modify or otherwise deal with
 * the Software or whether you are entitled to disclose the Software to any
 * other person you should contact PEGATRON.  If you have not entered into a
 * confidentiality agreement with PEGATRON granting access to this Software you
 * should forthwith return all media, copies and printed listings containing
 * the Software to PEGATRON.
 *
 * PEGATRON reserves the right to take legal action against you should you breach
 * the above provisions.
 *
 ******************************************************************************/

#include <string>

#include <boost/filesystem.hpp>

#include <stdio.h>

namespace fs = ::boost::filesystem;

using namespace std;

// ----------------------------------------------------------------------------------------

std::string strip_path (
    const std::string& str,
    const std::string& prefix
);
/*!
    ensures
        - if (prefix is a prefix of str) then
            - returns the part of str after the prefix
              (additionally, str will not begin with a / or \ character)
        - else
            - return str
!*/

// ----------------------------------------------------------------------------------------

void make_empty_file (
    const std::string& filename
);
/*!
    ensures
        - creates an empty file of the given name
!*/

// ----------------------------------------------------------------------------------------

void get_files_in_directory(const fs::path& root, const string& ext, std::vector<std::string>& files);
