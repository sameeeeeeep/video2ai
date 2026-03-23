class Video2ai < Formula
  include Language::Python::Virtualenv

  desc "Convert video to AI-ingestable format (frames + transcript + LLM analysis)"
  homepage "https://github.com/sameeeeeeep/video2ai"
  url "https://github.com/sameeeeeeep/video2ai/archive/refs/tags/v0.1.0.tar.gz"
  sha256 ""  # Update after first release: shasum -a 256 v0.1.0.tar.gz
  license "MIT"

  depends_on "python@3.12"
  depends_on "ffmpeg"

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "video2ai", shell_output("#{bin}/video2ai --version")
  end
end
