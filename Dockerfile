ARG VLLM_VERSION=v0.10.2
FROM vllm/vllm-openai:${VLLM_VERSION}

WORKDIR /app
COPY scripts/launch_vllm.sh /usr/local/bin/launch-vllm
RUN chmod 0755 /usr/local/bin/launch-vllm

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/launch-vllm"]
