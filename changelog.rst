Changes after 1.05
..................

* handling of cache files (problem solved: user control of location for large files from bigger models):

  * not strictly limited to *symenergy.cache* any longer
  * the cache path can be set using prior to the initialization of `Model` and/or `Evaluator`
  
  ::
      
      from symenergy import cache_params
      cache_params['path'] = os.path.abspath('./symenergy_cache')


  
  
