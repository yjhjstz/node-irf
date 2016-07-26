{
  "targets": [
    {
      "target_name": "irf",
      "sources": [
        "irf/node.cpp",
        "irf/randomForest.h",
        "irf/randomForest.cpp",
        "irf/MurmurHash3.h",
        "irf/MurmurHash3.cpp"
      ],
      "include_dirs" : [
        "<!(node -e \"require('nan')\")"
      ],
      'conditions': [
        [ 'OS=="linux" or OS=="freebsd" or OS=="openbsd" or OS=="solaris"', {
          'cflags_cc!': ['-fno-rtti', '-fno-exceptions'],
          'cflags_cc+': ['-frtti', '-fexceptions'],
        }],
        ['OS=="mac"', {
          'xcode_settings': {
            "OTHER_CPLUSPLUSFLAGS" : ["-std=c++11","-mmacosx-version-min=10.5"],
            "OTHER_LDFLAGS": ["-std=c++11"],
            'GCC_ENABLE_CPP_RTTI': 'YES',
            'GCC_ENABLE_CPP_EXCEPTIONS': 'YES'
          }
        }]
      ]
    }
  ]
}
