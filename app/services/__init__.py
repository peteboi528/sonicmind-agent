"""服务层：从 AudioVisualAgent 抽离的各领域编排服务。

每个 service 通过构造注入依赖与回调（不持 agent 引用），agent 侧保留同名薄委托。
- recommend / library / search / playlist / journey / discover / playback /
  catalog / taste_experiment / profile / tools
"""
