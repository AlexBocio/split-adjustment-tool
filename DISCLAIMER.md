# Disclaimer

**tapetruth is provided for educational and research purposes only.**

## No warranty

THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE, TITLE, AND NON-INFRINGEMENT. THE AUTHORS AND CONTRIBUTORS MAKE NO REPRESENTATION
OR WARRANTY THAT THE SOFTWARE WILL DETECT, REPAIR, OR CORRECTLY CLASSIFY ANY PARTICULAR
DEFECT IN ANY DATASET, OR THAT ITS OUTPUT IS ACCURATE, COMPLETE, OR SUITABLE FOR ANY
PURPOSE.

## Limitation of liability

IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE, INCLUDING
WITHOUT LIMITATION ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION), HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. SEE THE APACHE LICENSE,
VERSION 2.0 (`LICENSE`), SECTIONS 7 AND 8, FOR THE FULL, GOVERNING WARRANTY AND LIABILITY
TERMS.

## Not investment advice

tapetruth IS A DATA-QUALITY TOOL, NOT AN INVESTMENT ADVISOR. NOTHING IN THIS SOFTWARE, ITS
DOCUMENTATION, ITS OUTPUT (INCLUDING ANY REPORT, SCORECARD, RECONCILIATION RESULT, OR
"CERTIFICATION"), OR ANY ACCOMPANYING MATERIAL CONSTITUTES OR SHOULD BE CONSTRUED AS
INVESTMENT, FINANCIAL, LEGAL, OR TAX ADVICE, OR A RECOMMENDATION TO BUY, SELL, OR HOLD ANY
SECURITY OR FINANCIAL INSTRUMENT. tapetruth does not have access to, and does not
distribute, any licensed market data — it operates only on data you supply yourself, and it
tells you nothing about whether a security is a good or bad investment.

## Scope

tapetruth's checks and repairs are heuristic, tolerance-based, and — as documented in
`docs/STANDARD.md` §8 ("Known gaps") — known to be incomplete. A clean scorecard on the
included synthetic gauntlet (`python -m tapetruth.demo`) is evidence the engine is working
as designed against a known, planted set of defects; it is not a guarantee that your own
data is free of any defect, including defect classes the gauntlet does not model. Always
validate output against your own independent sources before relying on it for anything
consequential.

## Purpose statement

This project is published for educational and research purposes: to document a corporate-
action and OHLCV bar-data verification method and provide a reference, tape-truth-based
implementation. It is not offered as a commercial product or service. No fee, subscription,
or other consideration is charged for the software described in this repository.

---

*For the full license terms (including the ALL-CAPS warranty disclaimer in Sections 7–8),
see [`LICENSE`](LICENSE) (Apache License, Version 2.0).*
