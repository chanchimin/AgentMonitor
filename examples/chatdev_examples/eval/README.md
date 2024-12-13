## Evaluation

### Metrics

1. **Completeness** measures the software’s ability to fulfill code completion in software development, quantified as the percentage of software without any "placeholder" code snippets. A higher score indicates a higher probability of automated completion.

2. **Executability** assesses the software’s ability to run correctly within a compilation environment quantified as the percentage of software that compiles successfully and can run directly. A higher score indicates a higher probability of successful execution.

3. **Consistency** measures how closely the generated software code aligns with the original requirement description, quantified as the cosine distance between the semantic embeddings of the textual requirements and the generated software code. A higher score indicates a greater degree of consistency with the requirements.

4. **Quality** is a comprehensive metric that integrates various factors to assess the overall quality of software, quantified by multiplying completeness, executability, and consistency. A higher quality score suggests a higher overall satisfaction with the software generated, implying a lower need for further manual intervention.

### Usage

~~~bash
python generate_autonomy.py [path_to_WareHouse]
~~~

> NOTE: provide openai-key in `analysis_utils.py` and `generate_autonomy.py` first.
