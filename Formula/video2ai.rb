class Video2ai < Formula
  include Language::Python::Virtualenv

  desc "Convert video to AI-ingestable format (frames + transcript + LLM analysis)"
  homepage "https://github.com/sameeeeeeep/video2ai"
  url "https://github.com/sameeeeeeep/video2ai/archive/refs/tags/v0.1.1.tar.gz"
  sha256 "82c557f770bb1a41fec106b4a3ef9770653cab305ecc1a02d3163c2f1c6e2d78"  # Update after first release: shasum -a 256 v0.1.0.tar.gz
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
