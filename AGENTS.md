# 学习内容与维护约定

中文连续讲解为主，不要分点介绍。每天围绕恰好三个相互关联的 vLLM 设计问题，通常每篇约 1000—1800 字，篇幅服从讲透需要。以一个请求贯穿系统位置、真实矛盾、自然方案的限制、关键洞察、状态变化、不变量、故障反例和取舍。映射、共享或并发关系配机制图，注明教学示意，并完整解释图中关系。机制图须实际渲染核验。

开始时阅读 README.md、ROADMAP.md、index.json 和最近两天正文，以 Asia/Shanghai 日期归档并检查主题是否重复。同一天只有一组文章；重试补齐已有三篇，正文完整而发布失败时只重试发布。同一模块可继续深入新的问题，不以目录介绍或术语解释凑数。

通过 VLLM_SOURCE_DIR 指定只读源码 checkout，默认使用同级的 vllm 目录。固定完整源码提交，用 git show COMMIT:path 核对主论证；不修改、切换或拉取上游源码。未提交改动不得当作固定提交的代码。链接使用上游完整提交 permalink 与准确行号；外部事实查官方文档、原始论文或上游代码。

区分静态源码事实、教学简化、运行观察与性能测量。未运行的实验只给断点、状态变量、预期观察和能推翻解释的结果，不将假想轨迹或论文数字当成当前性能。不要为静态导读安装模型环境或启动、租用 GPU。

每天三个 Markdown 文件放在 daily/YYYY-MM-DD/，同步 README 阅读归档与 index.json。索引包含 date、topic_id、title、file、source_commit、evidence、status、sources；sources 包含 path、symbol、line。完成核验后才标 complete。

所有 Python 命令使用由 uv 创建的 .venv/bin/python，禁止系统 Python 或裸 pip。核验与发布只依赖标准库，无需安装 vLLM。环境已经建立时复用；需要重建时用 uv venv --python 3.12 .venv，仓库内已有忽略的 .tools/uv 也可用于创建环境。

公开文件不要包含个人姓名、私人邮箱、本机用户名、绝对目录、私有主机、内部聊天标识、凭据、个人日程或自动化审批记录。技术所需的公开上游引用、相对源码路径和源码提交可以保留。源码目录来自环境变量或同级相对路径，发布目标从 origin 识别。提交前运行 .venv/bin/python scripts/check_privacy.py --history，并人工检查正文语境。

发布执行 .venv/bin/python scripts/publish.py YYYY-MM-DD，仅暂存当天正文、README.md 和 index.json。保留用户更改，禁止 git add .、强推、reset 或清理用户文件。工具维护单独提交。发布器通过 GitHub CLI 现有认证推送，不读取或打印令牌；远端 main 与本地 HEAD 相同才报告 github_verified: true。失败保留正文和本地提交并如实说明。
