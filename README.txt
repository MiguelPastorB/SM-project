Error de too many request de Gemini por demasiadas consultas: (no pongas todo que son datos personales también)

Traceback (most recent call last):
  File "C:\Repositorios\sm-project\SM\Lib\site-packages\agno\models\google\gemini.py", line 343, in invoke_stream
    for response in self.get_client().models.generate_content_stream(
  File "C:\Repositorios\sm-project\SM\Lib\site-packages\google\genai\models.py", line 5364, in generate_content_stream
    yield from self._generate_content_stream(
  File "C:\Repositorios\sm-project\SM\Lib\site-packages\google\genai\models.py", line 4089, in _generate_content_stream
    for response in self._api_client.request_streamed(
  File "C:\Repositorios\sm-project\SM\Lib\site-packages\google\genai\_api_client.py", line 1405, in request_streamed
    session_response = self._request(http_request, http_options, stream=True)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Repositorios\sm-project\SM\Lib\site-packages\google\genai\_api_client.py", line 1224, in _request
    return self._retry(self._request_once, http_request, stream)  # type: ignore[no-any-return]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Repositorios\sm-project\SM\Lib\site-packages\tenacity\__init__.py", line 477, in __call__
    do = self.iter(retry_state=retry_state)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Repositorios\sm-project\SM\Lib\site-packages\tenacity\__init__.py", line 378, in iter
    result = action(retry_state)
             ^^^^^^^^^^^^^^^^^^^
  File "C:\Repositorios\sm-project\SM\Lib\site-packages\tenacity\__init__.py", line 420, in exc_check
    raise retry_exc.reraise()
          ^^^^^^^^^^^^^^^^^^^
  File "C:\Repositorios\sm-project\SM\Lib\site-packages\tenacity\__init__.py", line 187, in reraise
    raise self.last_attempt.result()
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Miguel\AppData\Local\Programs\Python\Python312\Lib\concurrent\futures\_base.py", line 449, in result
    return self.__get_result()
           ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Miguel\AppData\Local\Programs\Python\Python312\Lib\concurrent\futures\_base.py", line 401, in __get_result
    raise self._exception
  File "C:\Repositorios\sm-project\SM\Lib\site-packages\tenacity\__init__.py", line 480, in __call__
    result = fn(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^
  File "C:\Repositorios\sm-project\SM\Lib\site-packages\google\genai\_api_client.py", line 1189, in _request_once
    errors.APIError.raise_for_response(response)
  File "C:\Repositorios\sm-project\SM\Lib\site-packages\google\genai\errors.py", line 106, in raise_for_response
    cls.raise_error(response.status_code, response_json, response)
  File "C:\Repositorios\sm-project\SM\Lib\site-packages\google\genai\errors.py", line 131, in raise_error
    raise ClientError(status_code, response_json, response)
google.genai.errors.ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/usage?tab=rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 10, model: gemini-2.5-flash\nPlease retry in 18.186484543s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash'}, 'quotaValue': '10'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '18s'}]}}

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Repositorios\sm-project\proyecto_SM\main.py", line 96, in <module>
    main()
  File "C:\Repositorios\sm-project\proyecto_SM\main.py", line 89, in main
    modeling_agent.print_response(f"Divide los datos entre train y test del archivo {archivo_actual}."
  File "C:\Repositorios\sm-project\SM\Lib\site-packages\agno\agent\agent.py", line 9962, in print_response
    print_response_stream(
  File "C:\Repositorios\sm-project\SM\Lib\site-packages\agno\utils\print_response\agent.py", line 84, in print_response_stream
    for response_event in agent.run(
  File "C:\Repositorios\sm-project\SM\Lib\site-packages\agno\agent\agent.py", line 1271, in _run_stream
    for event in self._handle_model_response_stream(
  File "C:\Repositorios\sm-project\SM\Lib\site-packages\agno\agent\agent.py", line 4779, in _handle_model_response_stream
    for model_response_event in self.model.response_stream(
  File "C:\Repositorios\sm-project\SM\Lib\site-packages\agno\models\base.py", line 931, in response_stream
    for response in self.process_response_stream(
  File "C:\Repositorios\sm-project\SM\Lib\site-packages\agno\models\base.py", line 863, in process_response_stream
    for response_delta in self.invoke_stream(
  File "C:\Repositorios\sm-project\SM\Lib\site-packages\agno\models\google\gemini.py", line 354, in invoke_stream
    raise ModelProviderError(
agno.exceptions.ModelProviderError: <Response [429 Too Many Requests]>
